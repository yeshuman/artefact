# Generated manually

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('entities', '0002_initial'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='ReferenceEntity',
            new_name='EntityReference',
        ),
    ] 