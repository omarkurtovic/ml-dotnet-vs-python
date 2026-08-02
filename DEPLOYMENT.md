# Deployment Guide (Hetzner VPS)

Target: Ubuntu 24.04, CX23 (2 vCPU, 4 GB RAM)

## Prerequisites (one-time server setup)

SSH into the server:
```sh
ssh root@<your-server-ip>
```

Install system dependencies:
```sh
apt-get update
apt-get install -y dotnet-sdk-10.0 python3 python3-pip python3-venv git nginx
# Required by opencv-python
apt-get install -y libgl1
```

Verify:
```sh
dotnet --version   # should print 10.x.x
python3 --version  # should print 3.x.x
```

---

## 1. Clone the Repo

```sh
mkdir -p /opt/app
cd /opt/app
git clone https://github.com/omarkurtovic/ml-dotnet-vs-python
cd ml-dotnet-vs-python
```

> The repo will be at `/opt/app/ml-dotnet-vs-python/`. All subsequent server commands assume this path.

---

## 2. Transfer Data, Models

```powershell
# data and models folder
scp -r C:\Users\Administrator\source\repos\omarkurtovic\ml-dotnet-vs-python\storage root@<ip>:/opt/app/storage

```

---

## 3. Set Up Python

```sh
cd /opt/app/ml-dotnet-vs-python/python-model-trainer
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

---

## 4. Publish .NET Projects

```sh
cd /opt/app/ml-dotnet-vs-python
dotnet publish CSharpModelTrainerApi/CSharpModelTrainerApi.csproj -c Release -r linux-x64 --self-contained false -o /opt/app/publish/api
dotnet publish WebApp/WebApp.csproj -c Release -o /opt/app/publish/web
```

## 5. Create Configuration Files (on server)

### CSharpModelTrainerApi - tell it where the Python API is running and where to store models/data
```sh
cat > /opt/app/publish/api/appsettings.json << 'EOF'
{
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Microsoft.AspNetCore": "Warning"
    }
  },
  "Storage": {
    "Root": "/opt/app/storage"
  },
  "Services": {
    "pythonapi": {
      "http": ["http://localhost:8000"]
    }
  },
  "AllowedHosts": "*"
}
EOF
```

### WebApp — tells it where the C# API and Python API are running
```sh
cat > /opt/app/publish/web/appsettings.json << 'EOF'
{
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Microsoft.AspNetCore": "Warning"
    }
  },
  "AllowedHosts": "*",
  "Services": {
    "apiservice": {
      "http": ["http://localhost:5000"]
    },
    "pythonapi": {
      "http": ["http://localhost:8000"]
    }
  }
}
EOF
```

---

## 6. Create systemd Services (on server)

### C# API service
```sh
cat > /etc/systemd/system/ml-api.service << 'EOF'
[Unit]
Description=ML CSharp API
After=network.target

[Service]
WorkingDirectory=/opt/app/publish/api
ExecStart=/usr/bin/dotnet /opt/app/publish/api/CSharpModelTrainerApi.dll
Restart=always
RestartSec=10
Environment=ASPNETCORE_ENVIRONMENT=Production
Environment=ASPNETCORE_URLS=http://localhost:5000
Environment=ML_STORAGE_ROOT=/opt/app/storage

[Install]
WantedBy=multi-user.target
EOF
```

### Python API service
```sh
cat > /etc/systemd/system/ml-python.service << 'EOF'
[Unit]
Description=ML Python API
After=network.target

[Service]
WorkingDirectory=/opt/app/ml-dotnet-vs-python/python-model-trainer
ExecStart=/opt/app/ml-dotnet-vs-python/python-model-trainer/.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=10
Environment=ML_STORAGE_ROOT=/opt/app/storage

[Install]
WantedBy=multi-user.target
EOF
```

### Blazor WebApp service
```sh
cat > /etc/systemd/system/ml-web.service << 'EOF'
[Unit]
Description=ML Blazor WebApp
After=network.target ml-api.service

[Service]
WorkingDirectory=/opt/app/publish/web
ExecStart=/usr/bin/dotnet /opt/app/publish/web/WebApp.dll
Restart=always
RestartSec=10
Environment=ASPNETCORE_ENVIRONMENT=Production
Environment=ASPNETCORE_URLS=http://localhost:5001

[Install]
WantedBy=multi-user.target
EOF
```

Enable and start all services:
```sh
systemctl daemon-reload
systemctl enable ml-api ml-python ml-web
systemctl start ml-api ml-python ml-web
```

Check they are running:
```sh
systemctl status ml-api ml-python ml-web
```

---

## 7. Configure Nginx

First, install Nginx if not already installed:
```sh
apt-get install -y nginx
```

Then create the config:
```sh
cat > /etc/nginx/sites-available/ml-app << 'EOF'
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

server {
    listen 80;
    server_name _;

    location / {
        proxy_pass         http://localhost:5001;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection $connection_upgrade;
        proxy_set_header   Host $host;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 100s;
    }
}
EOF

ln -s /etc/nginx/sites-available/ml-app /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx
```

The app should now be accessible at `http://<your-server-ip>`.

---


## Updating the App

```sh
cd /opt/app/ml-dotnet-vs-python
git pull
dotnet publish CSharpModelTrainerApi/CSharpModelTrainerApi.csproj -c Release -r linux-x64 --self-contained false -o /opt/app/publish/api
dotnet publish WebApp/WebApp.csproj -c Release -o /opt/app/publish/web
systemctl restart ml-api ml-web
```

For Python:
```sh
systemctl restart ml-python
```

To sync just the models and database (not the dataset in `storage/data`):
```powershell
scp C:\Users\Administrator\source\repos\omarkurtovic\ml-dotnet-vs-python\storage\app.db root@<ip>:/opt/app/storage/app.db
scp -r C:\Users\Administrator\source\repos\omarkurtovic\ml-dotnet-vs-python\storage\models root@<ip>:/opt/app/storage/
systemctl restart ml-api
```

---

## Checking Logs

```sh
journalctl -u ml-api -f
journalctl -u ml-python -f
journalctl -u ml-web -f
```

---

## Troubleshooting

### Edited a systemd unit file? Run `daemon-reload` before restarting

systemd caches unit files at load time. If you hand-edit `/etc/systemd/system/ml-api.service` (or any other unit) — e.g. to change `ML_STORAGE_ROOT` — a plain `systemctl restart ml-api` reuses the *previously loaded* definition, env vars included. The service comes back up fine, with no error, just with the old (or no) environment variable. You must reload first:

```sh
systemctl daemon-reload
systemctl restart ml-api
```

### Models/data not showing up in the app

The C# API creates an empty SQLite DB with an applied schema wherever `ML_STORAGE_ROOT` happens to resolve to, if nothing exists there yet (`db.Database.Migrate()` runs unconditionally at startup). So a misconfigured storage path doesn't throw — it just silently serves an empty database. To confirm what a running service is actually using:

```sh
# What env var does the process actually have?
systemctl show ml-api -p Environment

# Which app.db file does it actually have open?
lsof -p $(pgrep -f CSharpModelTrainerApi.dll) | grep app.db

# Does the DB on disk actually have rows?
sqlite3 /opt/app/storage/app.db "SELECT COUNT(*) FROM LCModels;"
```

If the WebApp shows no models but `curl -s http://localhost:5000/LungCancer/Models` (run on the server) returns your data, the problem is downstream of the API — check the WebApp's `appsettings.json` `Services:apiservice`/`Services:pythonapi` URLs and `journalctl -u ml-web` instead.

---

## Port Summary

| Service              | Port  | Exposed |
|----------------------|-------|---------|
| CSharpModelTrainerApi | 5000 | Internal only |
| Python FastAPI        | 8000 | Internal only |
| Blazor WebApp         | 5001 | Via Nginx on port 80 |
| Nginx                 | 80   | Public |
