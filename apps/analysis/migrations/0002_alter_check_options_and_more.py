from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [("analysis", "0001_initial")]
    operations = [
        migrations.AlterModelOptions(name="check", options={"ordering": ["-created_at"]}),
        migrations.RenameField(model_name="detectedfeature", old_name="check", new_name="analysis_check"),
        migrations.RenameField(model_name="urlresult", old_name="check", new_name="analysis_check"),
        migrations.RenameIndex(model_name="check", old_name="analysis_c_user_id_8b6f5a_idx", new_name="analysis_ch_user_id_61cb17_idx"),
        migrations.RenameIndex(model_name="check", old_name="analysis_c_risk_le_0c7d6b_idx", new_name="analysis_ch_risk_le_783805_idx"),
    ]
