import os
from typing import Optional
import assets
import pygame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_font(size: int, font_name: Optional[str] = "Alegreya-Regular.otf") -> pygame.font.Font:
    """
    Utility to load a font of the given size.
    If font_name is provided, it tries to load that .ttf/.otf file from an 'assets/' subdirectory.
    Otherwise, it falls back to the default system font.
    """
    if font_name:
        # Construct the full path to the font file
        # Assumes your utils.py is at the root or you adjust the path accordingly
        font_path = os.path.join(BASE_DIR, "assets", font_name)
        try:
            return pygame.font.Font(font_path, size)
        except pygame.error as e:
            print(f"Warning: Could not load custom font '{font_path}': {e}. Falling back to default font.")

    # Fallback to default font if no name provided or custom font fails to load
    return pygame.font.Font(pygame.font.get_default_font(), size)


def draw_text_centered(surface, text, font, color, center):
    """
    Draw text on the given surface, centered at `center` (x, y).
    """
    rendered = font.render(text, True, color)
    rect = rendered.get_rect()
    rect.center = center
    surface.blit(rendered, rect)
