from django.core.management.base import BaseCommand
from dojo.dojo import Dojo
from openai import AsyncOpenAI
import asyncio
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Run an autonomous dojo session'

    def add_arguments(self, parser):
        parser.add_argument(
            '--max-exchanges',
            type=int,
            default=5,
            help='Maximum number of message exchanges'
        )

    def handle(self, *args, **options):
        """Sync wrapper for async handle method."""
        asyncio.run(self.async_handle(*args, **options))

    async def async_handle(self, *args, **options):
        """Handle the command execution."""
        try:
            max_exchanges = options['max_exchanges']
            logger.info(f"Starting autonomous dojo session with max {max_exchanges} exchanges")
            
            # Initialize the dojo
            dojo = Dojo()
            await dojo.initialize()
            logger.info("Dojo initialized successfully")
            
            # Prepare participants
            await dojo.prepare_ronin()
            await dojo.prepare_satori()
            logger.info("Participants prepared successfully")
            
            # Create mondo and start conversation
            mondo = await dojo.mondo()
            logger.info(f"Created mondo {mondo.id}")
            
            # Process exchanges
            exchanges_count = 0
            while exchanges_count < max_exchanges:
                try:
                    should_continue = await dojo.process_exchange(mondo)
                    if not should_continue:
                        logger.info("Conversation reached natural conclusion")
                        break
                        
                    exchanges_count += 1
                    logger.info(f"Completed exchange {exchanges_count}/{max_exchanges}")
                    
                except Exception as e:
                    logger.error(f"Error processing exchange: {str(e)}")
                    break
                    
            logger.info(f"Dojo session completed with {exchanges_count} exchanges")
            
        except Exception as e:
            logger.error(f"Error running dojo: {str(e)}")
            raise 