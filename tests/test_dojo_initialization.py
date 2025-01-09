import pytest
from unittest.mock import patch
from dojo.dojo import Dojo
from entities.models import Entity, EntityArchetype, EntityReference
from tests.test_dojo import MockOpenAI

@pytest.mark.django_db(transaction=True)
@pytest.mark.mock
async def test_dojo_initialization(mock_llm_responses):
    """Test that a Dojo is initialized with theme, principles, and entity archetypes."""
    
    # Reset mock state
    from tests.test_dojo import state
    state['message_count'] = 0
    state['exchange_count'] = 0
    state['last_speaker'] = None
    state['responses'] = mock_llm_responses
    
    with patch('openai.AsyncOpenAI', return_value=MockOpenAI()):
        # Create a dojo instance with mock client
        dojo = Dojo(llm_client=MockOpenAI())
        
        try:
            # Initialize the dojo
            await dojo.initialize()
            
            # Verify dojo model was created
            assert dojo.model is not None
            assert dojo.model.theme != ""
            assert len(dojo.model.principles) > 0
            assert dojo.model.ronin_system_message != ""
            assert dojo.model.satori_system_message != ""
            
            # Verify entity archetypes were created
            archetypes = await EntityArchetype.objects.all().acount()
            assert archetypes > 0, "No entity archetypes were created"
            
            # Get first archetype and verify its structure
            first_archetype = await EntityArchetype.objects.all().afirst()
            assert first_archetype.name != ""
            assert first_archetype.description != ""
            assert first_archetype.embedding is not None
            assert len(first_archetype.embedding) == 1536  # OpenAI embedding size
            
            # Verify reference entities were created
            references = await EntityReference.objects.all().acount()
            assert references > 0, "No reference entities were created"
            
            # Get first reference and verify its structure
            first_reference = await EntityReference.objects.select_related('archetype').all().afirst()
            assert first_reference.text != ""
            assert first_reference.archetype is not None
            assert first_reference.embedding is not None
            assert len(first_reference.embedding) == 1536
            
            # Log the created archetypes and references for inspection
            archetypes = []
            async for archetype in EntityArchetype.objects.values('name', 'description'):
                archetypes.append(archetype)
            print("\nCreated Archetypes:")
            for archetype in archetypes:
                print(f"- {archetype['name']}: {archetype['description']}")
                
            references = []
            async for ref in EntityReference.objects.select_related('archetype').values('text', 'archetype__name'):
                references.append(ref)
            print("\nCreated References:")
            for ref in references:
                print(f"- {ref['text']} ({ref['archetype__name']})")
                
        finally:
            # Cleanup
            await dojo.cleanup() 