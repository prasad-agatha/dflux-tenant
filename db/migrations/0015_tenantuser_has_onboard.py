# Generated by Django 3.1.5 on 2023-02-02 18:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0014_auto_20230202_1151'),
    ]

    operations = [
        migrations.AddField(
            model_name='tenantuser',
            name='has_onboard',
            field=models.BooleanField(default=False),
        ),
    ]
