
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

def play_sound(soundpath):
    pygame.mixer.init()
    pygame.mixer.music.load(soundpath)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue

class PyRootsError(Exception):
    pygame.mixer.init()
    pygame.mixer.music.load('src/exceptions/smb_gameover.wav')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue
