# Generated manually

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('mondos', '0001_initial'),
        ('entities', '0003_rename_referenceentity_entityreference'),
    ]

    operations = [
        migrations.AlterField(
            model_name='entityreference',
            name='mondo',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='entity_references', to='mondos.mondo'),
        ),
    ] 