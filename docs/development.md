# Development Guidelines

## Database Migrations

### Early Development
During early development phases when the database schema is still evolving rapidly:

1. Use `reset.sh` to reset the database and recreate it from scratch
2. This is preferable to creating multiple migrations that may conflict
3. Only start maintaining proper migrations once the schema is more stable

**Note**: `reset.sh` should ONLY be used in development, never in production.

### Production Migrations
Once the schema stabilizes:

1. Create proper migrations using `python manage.py makemigrations`
2. Test migrations thoroughly before deploying
3. Never use `reset.sh` in production environments
4. Maintain backwards compatibility in migrations 