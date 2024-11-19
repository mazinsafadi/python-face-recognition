az container create `
  --resource-group mazin `
  --name python-face-recognition-container `
  --image mazin.azurecr.io/python-face-recognition:latest `
  --registry-login-server mazin.azurecr.io `
  --registry-username <your-user-name> `
  --registry-password <your-registry-password> `
  --dns-name-label python-face-recognition `
  --ports 80 `
  --os-type Linux `
  --cpu 2 `
  --memory 4 `
  --restart-policy OnFailure
