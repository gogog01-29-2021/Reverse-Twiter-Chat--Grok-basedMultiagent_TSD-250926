# Multi-Agent CAE Platform v0.2

A multi-agent Computer-Aided Engineering (CAE) simulation platform with AI-powered workflow design using Google Gemini.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Google Gemini API key

### 1. Setup API Key
Insert your Google Gemini API key in ./backend/.env file:
```bash
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=your_actual_api_key_here
```

### 2. Start the Application

**Option A: Using the provided script (Linux/WSL)**
```bash
./bin/run.sh start
```

**Option B: Direct Docker Compose (Windows/Cross-platform)**
```bash
docker compose up -d --build
```

### 3. Access the Application
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🛑 How to Stop Safely

### Option A: Using the script
```bash
./bin/run.sh stop
```

### Option B: Direct Docker Compose
```bash
docker compose down
```

### What happens when you stop:
- ✅ All containers will be stopped gracefully
- ✅ Network connections will be closed
- ✅ **Your data is preserved** - containers are stopped, not deleted
- ✅ You can restart anytime with the same data

### Force stop (if containers are unresponsive)
```bash
# Remove containers and networks (data preserved)
docker compose down --remove-orphans

# Remove everything including volumes (⚠️ data will be lost)
docker compose down --remove-orphans --volumes
```

### ⚠️ Important Notes:
- **Normal stop** (docker compose down): Containers stop but data is preserved
- **Force stop with volumes**: Will delete all data including database/file changes
- **Graceful shutdown**: Always use normal stop first, force stop only if needed

## 📊 Monitoring & Maintenance

### Check Status
```bash
# Using script
./bin/run.sh status

# Using Docker Compose
docker compose ps
```

### View Logs
```bash
# Using script
./bin/run.sh logs

# Using Docker Compose
docker compose logs -f

# View specific service logs
docker compose logs backend
docker compose logs frontend
```

### Restart Services
```bash
# Using script
./bin/run.sh restart

# Using Docker Compose
docker compose restart
```

## 🔧 Development

### Access Container Shells
```bash
# Backend container
./bin/run.sh shell-backend
# or
docker compose exec backend /bin/sh

# Frontend container
./bin/run.sh shell-frontend
# or
docker compose exec frontend /bin/sh
```

### Rebuild After Changes
```bash
docker compose up -d --build
```

## 🐛 Troubleshooting

### Port Conflicts
If ports 5173 or 8000 are already in use, modify the port mappings in docker-compose.yml:
```yaml
ports:
  - "8001:8000"  # Change host port
  - "3000:80"    # Change host port
```

### Container Issues
1. Check logs: docker compose logs [service_name]
2. Restart services: docker compose restart
3. Rebuild if needed: docker compose up -d --build
4. Clean restart: docker compose down && docker compose up -d --build

---

## Autonomous Information Crawling Vision

The Reverse-Twiter-Chat--Grok-basedMultiagent_TSD-250926 concept expands the platform into an AI agent that:
- Crawls designated browser tabs, histories, and logs (Chrome, Safari, etc.).
- Summarizes personal notes, chat histories (ChatGPT, Gemini, Kimi, ...), and local storage content.
- Stores summaries in a primary database, plans follow-up actions in a secondary database, and executes them autonomously.
- Remains interactive so you can guide, pause, or redirect the agent at any time.

### Example Workflow
1. **Crawl**: Scan browsers, notes, chat logs, and local files for new information.
2. **Summarize**: Store condensed insights in the primary DB.
3. **Plan**: Analyse the DB, create deeper-exploration plans, and log them in a secondary DB.
4. **Act**: Execute the plan to gather targeted information.
5. **Interact**: Present progress for feedback so the workflow stays aligned with your priorities.

_Keep this section updated as you implement the broader autonomous information-gathering roadmap._
