from django.db import models

class BasketballGame(models.Model):
    video = models.FileField(upload_to='videos/')
    name = models.CharField(max_length=100, default='basketball_game')
    corners = models.JSONField(default=dict)
    distortion_parameters = models.JSONField(default=dict)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Game {self.id}"
