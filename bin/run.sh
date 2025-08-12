#!/bin/bash

# ANSI 색상 코드 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 색상 없음

set -e  # 오류 발생 시 스크립트 종료.
set -u  # 변수 참조 시 오류 발생.
set -o pipefail  # 파이프 실패 시 스크립트 종료.

# 사용법을 출력하는 함수
usage() {
    echo -e "${YELLOW}Usage: $0 {start|stop|restart|logs|status|shell-backend|shell-frontend}${NC}"
    echo "Commands:"
    echo "  start          : Build and start services in the background."
    echo "  stop           : Stop and remove services."
    echo "  restart        : Restart all services."
    echo "  logs           : View output from containers."
    echo "  status         : List running containers."
    echo "  shell-backend  : Start a shell session in the backend container."
    echo "  shell-frontend : Start a shell session in the frontend container."
    exit 1
}

# 첫 번째 인자가 없으면 사용법 출력
if [ -z "${1:-}" ]; then
    usage
fi

# 명령어에 따라 분기 처리
case "$1" in
    start)
        echo -e "${GREEN}Starting all services...${NC}"
        docker compose up -d --build
        echo -e "${GREEN}Services started successfully.${NC}"
        echo -e "Frontend is available at ${YELLOW}http://localhost:80${NC}"
        echo -e "Backend API docs are available at ${YELLOW}http://localhost:8000/docs${NC}"
        ;;
    stop)
        echo -e "${RED}Stopping all services...${NC}"
        docker compose down
        echo -e "${RED}Services stopped.${NC}"
        ;;
    restart)
        echo -e "${YELLOW}Restarting all services...${NC}"
        docker compose down
        docker compose up -d --build
        echo -e "${GREEN}Services restarted successfully.${NC}"
        ;;
    logs)
        echo -e "${GREEN}Attaching to logs... (Press Ctrl+C to exit)${NC}"
        docker compose logs -f
        ;;
    status)
        echo -e "${GREEN}Current status of services:${NC}"
        docker compose ps
        ;;
    shell-backend)
        echo -e "${GREEN}Connecting to the backend container shell...${NC}"
        docker compose exec backend /bin/sh
        ;;
    shell-frontend)
        echo -e "${GREEN}Connecting to the frontend container shell...${NC}"
        docker compose exec frontend /bin/sh
        ;;
    *)
        usage
        ;;
esac

exit 0
