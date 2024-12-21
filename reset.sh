#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment variables from .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo -e "${RED}Error: .env file not found${NC}"
    exit 1
fi

# PostgreSQL connection string
PG_CONN="-h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER"

echo -e "${YELLOW}Removing migrations (except entities)...${NC}"
find . -path "*/migrations/*.py" -not -path "./entities/*" -not -name "__init__.py" -delete
find . -path "*/migrations/*.pyc" -not -path "./entities/*" -delete

echo -e "${YELLOW}Terminating existing database connections...${NC}"
PGPASSWORD=postgres psql -U postgres -h localhost -c "
SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname IN ('artefact', 'test_artefact', 'test_artefact_mock')
  AND pid <> pg_backend_pid();"

echo -e "${YELLOW}Dropping test databases...${NC}"
PGPASSWORD=postgres dropdb -U postgres -h localhost test_artefact --if-exists
PGPASSWORD=postgres dropdb -U postgres -h localhost test_artefact_mock --if-exists

echo -e "${YELLOW}Dropping main database...${NC}"
PGPASSWORD=postgres dropdb -U postgres -h localhost artefact --if-exists

echo -e "${YELLOW}Creating main database...${NC}"
PGPASSWORD=postgres createdb -U postgres -h localhost artefact

echo -e "${YELLOW}Making migrations...${NC}"
python manage.py makemigrations

echo -e "${YELLOW}Applying migrations...${NC}"
python manage.py migrate

echo -e "${GREEN}Reset complete!${NC}"
echo -e "${YELLOW}Note: You may need to recreate any initial data or superuser accounts.${NC}"

# Optional: Create superuser if DJANGO_SUPERUSER_* environment variables are set
if [ ! -z "$DJANGO_SUPERUSER_USERNAME" ] && [ ! -z "$DJANGO_SUPERUSER_EMAIL" ] && [ ! -z "$DJANGO_SUPERUSER_PASSWORD" ]; then
    echo -e "${YELLOW}Creating superuser...${NC}"
    python manage.py createsuperuser --noinput
fi

echo -e "${GREEN}Done!${NC}" 