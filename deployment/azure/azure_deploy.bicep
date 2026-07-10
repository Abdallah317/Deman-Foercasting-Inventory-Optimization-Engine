// ─────────────────────────────────────────────────────────────
// azure_deploy.bicep
// Milestone 3 – Azure Infrastructure as Code
//
// Deploys:
//   • Azure Container Registry (ACR)
//   • Azure Container Apps (ACA) — runs the FastAPI service
//   • Azure Log Analytics Workspace — monitoring
//   • Azure Application Insights — telemetry
// ─────────────────────────────────────────────────────────────

@description('Location for all resources')
param location string = resourceGroup().location

@description('Environment name (dev / staging / prod)')
param environment string = 'prod'

@description('Container image tag')
param imageTag string = 'latest'

// ── Variables ──────────────────────────────────────────────────────────────
var appName         = 'demand-forecasting'
var acrName         = '${appName}acr${uniqueString(resourceGroup().id)}'
var containerAppEnv = '${appName}-env-${environment}'
var containerAppName= '${appName}-api-${environment}'
var logWorkspace    = '${appName}-logs-${environment}'
var appInsightsName = '${appName}-insights-${environment}'


// ── Log Analytics Workspace ────────────────────────────────────────────────
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name:     logWorkspace
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}


// ── Application Insights ───────────────────────────────────────────────────
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name:     appInsightsName
  location: location
  kind:     'web'
  properties: {
    Application_Type:             'web'
    WorkspaceResourceId:          logAnalytics.id
    IngestionMode:                'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery:     'Enabled'
  }
}


// ── Azure Container Registry ───────────────────────────────────────────────
resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name:     acrName
  location: location
  sku: { name: 'Basic' }
  properties: {
    adminUserEnabled: true
  }
}


// ── Container Apps Environment ─────────────────────────────────────────────
resource containerEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name:     containerAppEnv
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey:  logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}


// ── Container App (FastAPI Service) ───────────────────────────────────────
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name:     containerAppName
  location: location
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      ingress: {
        external:   true
        targetPort: 8000
        transport:  'http'
        corsPolicy: {
          allowedOrigins: ['*']
          allowedMethods: ['GET', 'POST', 'OPTIONS']
          allowedHeaders: ['*']
        }
      }
      registries: [
        {
          server:            acr.properties.loginServer
          username:          acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name:  'acr-password'
          value: acr.listCredentials().passwords[0].value
        }
        {
          name:  'appinsights-key'
          value: appInsights.properties.InstrumentationKey
        }
      ]
    }
    template: {
      containers: [
        {
          name:  'demand-forecasting-api'
          image: '${acr.properties.loginServer}/${appName}:${imageTag}'
          resources: {
            cpu:    '1.0'
            memory: '2Gi'
          }
          env: [
            { name: 'ENVIRONMENT',            value: environment }
            { name: 'APPINSIGHTS_KEY',        secretRef: 'appinsights-key' }
            { name: 'PYTHONUNBUFFERED',        value: '1' }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path:   '/health'
                port:   8000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 10
              periodSeconds:       30
            }
            {
              type: 'Readiness'
              httpGet: {
                path:   '/health'
                port:   8000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 5
              periodSeconds:       10
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 5
        rules: [
          {
            name: 'http-scaling'
            http: { metadata: { concurrentRequests: '50' } }
          }
        ]
      }
    }
  }
}


// ── Outputs ────────────────────────────────────────────────────────────────
output apiUrl            string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output acrLoginServer    string = acr.properties.loginServer
output appInsightsKey    string = appInsights.properties.InstrumentationKey
output containerAppName  string = containerApp.name
