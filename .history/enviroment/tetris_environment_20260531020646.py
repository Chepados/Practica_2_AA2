import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
 
# Definición de las piezas de Tetris con sus rotaciones
raw_shapes = {
    'S': [['.....',
           '.....',
           '..00.',
           '.00..',
           '.....'],
          ['.....',
           '..0..',
           '..00.',
           '...0.',
           '.....']],
    'Z': [['.....',
           '.....',
           '.00..',
           '..00.',
           '.....'],
          ['.....',
           '..0..',
           '.00..',
           '.0...',
           '.....']],
    'I': [['.....',
           '..0..',
           '..0..',
           '..0..',
           '..0..'],
          ['.....',
           '0000.',
           '.....',
           '.....',
           '.....']],
    'O': [['.....',
           '.....',
           '.00..',
           '.00..',
           '.....']],
    'J': [['.....',
           '.0...',
           '.000.',
           '.....',
           '.....'],
          ['.....',
           '..00.',
           '..0..',
           '..0..',
           '.....'],
          ['.....',
           '.....',
           '.000.',
           '...0.',
           '.....'],
          ['.....',
           '..0..',
           '..0..',
           '.00..',
           '.....']],
    'L': [['.....',
           '...0.',
           '.000.',
           '.....',
           '.....'],
          ['.....',
           '..0..',
           '..0..',
           '..00.',
           '.....'],
          ['.....',
           '.....',
           '.000.',
           '.0...',
           '.....'],
          ['.....',
           '.00..',
           '..0..',
           '..0..',
           '.....']],
    'T': [['.....',
           '..0..',
           '.000.',
           '.....',
           '.....'],
          ['.....',
           '..0..',
           '..00.',
           '..0..',
           '.....'],
          ['.....',
           '.....',
           '.000.',
           '..0..',
           '.....'],
          ['.....',
           '..0..',
           '.00..',
           '..0..',
           '.....']]
}
 
# Convertir las formas a arrays numpy
pieces = {
    name: [
        np.array([[1 if ch == '0' else 0 for ch in row] for row in shape], dtype=np.uint8)
        for shape in shapes
    ]
    for name, shapes in raw_shapes.items()
}
 
PIECE_NAMES = list(pieces.keys())
 
# Colores para pygame (formato RGB)
COLORS = {
    'S': (0, 255, 0),      # Verde
    'Z': (255, 0, 0),      # Rojo
    'I': (0, 255, 255),    # Cyan
    'O': (255, 255, 0),    # Amarillo
    'J': (0, 0, 255),      # Azul
    'L': (255, 165, 0),    # Naranja
    'T': (128, 0, 128),    # Púrpura
    'background': (0, 0, 0),
    'grid': (40, 40, 40),
    'text': (255, 255, 255)
}
 
 
class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human", "rgb_array"]}
 
    def __init__(self, n=20, m=10, render_mode=None):
        super().__init__()
        self.n = n  # Altura del tablero
        self.m = m  # Ancho del tablero
        self.render_mode = render_mode
 
        # Espacios de acción: 0=rotar_izq, 1=rotar_der, 2=izq, 3=der, 4=bajar, 5=hard_drop
        self.action_space = spaces.Discrete(6)
 
        # Espacio de observación
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(self.n, self.m), dtype=np.uint8),
            "current_piece": spaces.Discrete(7),
            "next_piece": spaces.Discrete(7)
        })
 
        # Configuración de pygame
        self.cell_size = 30
        self.screen = None
        self.clock = None
        if render_mode == "human":
            pygame.init()
            self.screen_width = (self.m + 6) * self.cell_size
            self.screen_height = self.n * self.cell_size
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Tetris")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
 
        # Estado del juego
        self.board = None
        self.current_piece_name = None
        self.next_piece_name = None
        self.current_rotation = 0
        self.current_position = None
        self.score = 0
        self.lines_cleared = 0
 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reiniciar estado
        self.board = np.zeros((self.n, self.m), dtype=np.uint8)
        self.score = 0
        self.lines_cleared = 0
        
        # Seleccionar piezas iniciales
        self.current_piece_name = self.np_random.choice(PIECE_NAMES)
        self.next_piece_name = self.np_random.choice(PIECE_NAMES)
        self.current_rotation = 0
        self.current_position = [0, self.m // 2 - 2]
        
        # Verificar si la pieza inicial colisiona (game over inmediato)
        if self._check_collision():
            # No debería pasar en reset, pero por si acaso
            pass
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
 
    def _get_obs(self):
        """Obtener la observación actual"""
        return {
            "board": self.board.copy(),
            "current_piece": PIECE_NAMES.index(self.current_piece_name),
            "next_piece": PIECE_NAMES.index(self.next_piece_name)
        }
 
    def _get_info(self):
        """Información adicional"""
        return {
            "score": self.score,
            "lines_cleared": self.lines_cleared
        }
 
    def _get_current_piece(self):
        """Obtener la pieza actual con su rotación"""
        return pieces[self.current_piece_name][self.current_rotation]
 
    def _check_collision(self, offset_row=0, offset_col=0, rotation=None):
        """Verificar si hay colisión con la posición y rotación dadas"""
        if rotation is None:
            rotation = self.current_rotation
        
        piece = pieces[self.current_piece_name][rotation]
        test_row = self.current_position[0] + offset_row
        test_col = self.current_position[1] + offset_col
        
        for r in range(piece.shape[0]):
            for c in range(piece.shape[1]):
                if piece[r, c]:
                    board_r = test_row + r
                    board_c = test_col + c
                    
                    # Verificar límites del tablero
                    if board_r < 0:
                        continue  # Permitir que la pieza esté parcialmente arriba al inicio
                    if board_r >= self.n or board_c < 0 or board_c >= self.m:
                        return True
                    
                    # Verificar colisión con piezas colocadas
                    if self.board[board_r, board_c]:
                        return True
        
        return False
 
    def _place_piece(self):
        """Colocar la pieza actual en el tablero"""
        piece = self._get_current_piece()
        
        for r in range(piece.shape[0]):
            for c in range(piece.shape[1]):
                if piece[r, c]:
                    board_r = self.current_position[0] + r
                    board_c = self.current_position[1] + c
                    if 0 <= board_r < self.n and 0 <= board_c < self.m:
                        self.board[board_r, board_c] = 1
 
    def _clear_lines(self):
        """Eliminar líneas completas y retornar el número de líneas eliminadas"""
        lines_to_clear = []
        
        for r in range(self.n):
            if np.all(self.board[r, :]):
                lines_to_clear.append(r)
        
        if lines_to_clear:
            # Eliminar las líneas completas
            self.board = np.delete(self.board, lines_to_clear, axis=0)
            # Añadir líneas vacías arriba
            new_lines = np.zeros((len(lines_to_clear), self.m), dtype=np.uint8)
            self.board = np.vstack([new_lines, self.board])
        
        return len(lines_to_clear)
 
    def _spawn_new_piece(self):
        """Generar una nueva pieza"""
        self.current_piece_name = self.next_piece_name
        self.next_piece_name = self.np_random.choice(PIECE_NAMES)
        self.current_rotation = 0
        self.current_position = [0, self.m // 2 - 2]
 
    def step(self, action):
        """Ejecutar una acción en el entorno"""
        old_position = self.current_position.copy()
        old_rotation = self.current_rotation
        
        step_lines = 0
        
        # Procesar la acción
        if action == 0:  # Rotar antihorario
            new_rotation = (self.current_rotation - 1) % len(pieces[self.current_piece_name])
            if not self._check_collision(rotation=new_rotation):
                self.current_rotation = new_rotation
                
        elif action == 1:  # Rotar horario
            new_rotation = (self.current_rotation + 1) % len(pieces[self.current_piece_name])
            if not self._check_collision(rotation=new_rotation):
                self.current_rotation = new_rotation
                
        elif action == 2:  # Mover izquierda
            if not self._check_collision(offset_col=-1):
                self.current_position[1] -= 1
                
        elif action == 3:  # Mover derecha
            if not self._check_collision(offset_col=1):
                self.current_position[1] += 1
                
        elif action == 4:  # Bajar una fila
            if not self._check_collision(offset_row=1):
                self.current_position[0] += 1
            else:
                # La pieza toca el fondo o otra pieza
                self._place_piece()
                lines = self._clear_lines()
                step_lines = lines
                self.lines_cleared += lines
                self.score += lines * 100  # 100 puntos por línea
                
                # Generar nueva pieza
                self._spawn_new_piece()
                
                # Verificar game over
                if self._check_collision():
                    obs = self._get_obs()
                    info = self._get_info()
                    info['step_lines_cleared'] = step_lines
                    return obs, self.score, True, False, info
                    
        elif action == 5:  # Hard drop (caída rápida)
            while not self._check_collision(offset_row=1):
                self.current_position[0] += 1
            
            # Colocar pieza
            self._place_piece()
            lines = self._clear_lines()
            step_lines = lines
            self.lines_cleared += lines
            self.score += lines * 100
            
            # Generar nueva pieza
            self._spawn_new_piece()
            
            # Verificar game over
            if self._check_collision():
                obs = self._get_obs()
                info = self._get_info()
                info['step_lines_cleared'] = step_lines
                return obs, self.score, True, False, info
        
        obs = self._get_obs()
        reward = self.score  # El reward es el score acumulado
        terminated = False
        truncated = False
        info = self._get_info()
        info['step_lines_cleared'] = step_lines
        
        return obs, reward, terminated, truncated, info
 
    def render(self):
        """Renderizar el estado del juego"""
        if self.render_mode == "human":
            return self._render_pygame()
        elif self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
 
    def _render_ansi(self):
        """Renderizado en formato ASCII"""
        # Crear una copia del tablero con la pieza actual
        display_board = self.board.copy()
        piece = self._get_current_piece()
        
        for r in range(piece.shape[0]):
            for c in range(piece.shape[1]):
                if piece[r, c]:
                    board_r = self.current_position[0] + r
                    board_c = self.current_position[1] + c
                    if 0 <= board_r < self.n and 0 <= board_c < self.m:
                        display_board[board_r, board_c] = 2  # 2 para pieza actual
        
        # Crear representación en texto
        lines = []
        lines.append("┌" + "─" * self.m + "┐")
        
        for row in display_board:
            line = "│"
            for cell in row:
                if cell == 0:
                    line += " "
                elif cell == 1:
                    line += "█"
                elif cell == 2:
                    line += "●"
            line += "│"
            lines.append(line)
        
        lines.append("└" + "─" * self.m + "┘")
        lines.append(f"Score: {self.score} | Lines: {self.lines_cleared}")
        lines.append(f"Next: {self.next_piece_name}")
        
        output = "\n".join(lines)
        print(output)
        return output
 
    def _render_pygame(self):
        """Renderizado con Pygame"""
        if self.screen is None:
            return
        
        # Procesar los eventos de pygame para que no se congele (Not Responding)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(COLORS['background'])
        
        # Dibujar el tablero con piezas colocadas
        for r in range(self.n):
            for c in range(self.m):
                x = c * self.cell_size
                y = r * self.cell_size
                
                # Dibujar celda
                if self.board[r, c]:
                    pygame.draw.rect(self.screen, (150, 150, 150), 
                                   (x, y, self.cell_size, self.cell_size))
                    pygame.draw.rect(self.screen, COLORS['grid'], 
                                   (x, y, self.cell_size, self.cell_size), 1)
                else:
                    pygame.draw.rect(self.screen, COLORS['grid'], 
                                   (x, y, self.cell_size, self.cell_size), 1)
        
        # Dibujar la pieza actual
        piece = self._get_current_piece()
        color = COLORS[self.current_piece_name]
        
        for r in range(piece.shape[0]):
            for c in range(piece.shape[1]):
                if piece[r, c]:
                    board_r = self.current_position[0] + r
                    board_c = self.current_position[1] + c
                    if 0 <= board_r < self.n and 0 <= board_c < self.m:
                        x = board_c * self.cell_size
                        y = board_r * self.cell_size
                        pygame.draw.rect(self.screen, color, 
                                       (x, y, self.cell_size, self.cell_size))
                        pygame.draw.rect(self.screen, COLORS['grid'], 
                                       (x, y, self.cell_size, self.cell_size), 1)
        
        # Dibujar información lateral
        info_x = (self.m + 1) * self.cell_size
        
        # Score
        score_text = self.font.render(f"Score:", True, COLORS['text'])
        self.screen.blit(score_text, (info_x, 50))
        score_value = self.font.render(f"{self.score}", True, COLORS['text'])
        self.screen.blit(score_value, (info_x, 90))
        
        # Lines
        lines_text = self.font.render(f"Lines:", True, COLORS['text'])
        self.screen.blit(lines_text, (info_x, 150))
        lines_value = self.font.render(f"{self.lines_cleared}", True, COLORS['text'])
        self.screen.blit(lines_value, (info_x, 190))
        
        # Next piece
        next_text = self.font.render(f"Next:", True, COLORS['text'])
        self.screen.blit(next_text, (info_x, 250))
        
        next_piece = pieces[self.next_piece_name][0]
        next_color = COLORS[self.next_piece_name]
        for r in range(next_piece.shape[0]):
            for c in range(next_piece.shape[1]):
                if next_piece[r, c]:
                    x = info_x + c * 20
                    y = 290 + r * 20
                    pygame.draw.rect(self.screen, next_color, (x, y, 20, 20))
                    pygame.draw.rect(self.screen, COLORS['grid'], (x, y, 20, 20), 1)
        
        pygame.display.flip()
        
        if self.clock:
            self.clock.tick(60)
 
    def _render_rgb_array(self):
        """Renderizado como array RGB (para grabación)"""
        # Similar a pygame pero retornando un array
        pass
 
    def close(self):
        """Cerrar el entorno"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
 
 
def main():
    """Demo interactiva del juego"""
    print("=== TETRIS DEMO ===")
    print("\nControles:")
    print("  0: Rotar antihorario")
    print("  1: Rotar horario")
    print("  2: Mover izquierda")
    print("  3: Mover derecha")
    print("  4: Bajar")
    print("  5: Hard drop (caída rápida)")
    print("\n¿Qué modo de renderizado prefieres?")
    print("  1: Pygame (gráfico)")
    print("  2: ASCII (texto)")
    
    mode_choice = input("Selecciona (1 o 2): ").strip()
    render_mode = "human" if mode_choice == "1" else "ansi"
    
    # Crear el entorno
    env = TetrisEnv(n=20, m=10, render_mode=render_mode)
    
    obs, info = env.reset()
    env.render()
    
    total_reward = 0
    step_count = 0
    
    print("\n¡Juego iniciado! Introduce acciones (0-5) o 'q' para salir.")
    
    while True:
        # Manejar eventos de pygame si está activo
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    sys.exit()
        
        # Obtener acción del usuario
        try:
            action_input = input("\nAcción: ").strip().lower()
            
            if action_input == 'q':
                print("¡Saliendo del juego!")
                break
            
            action = int(action_input)
            
            if action not in range(6):
                print("Acción inválida. Usa 0-5.")
                continue
            
            # Ejecutar la acción
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward = reward
            step_count += 1
            
            # Renderizar
            env.render()
            
            # Mostrar información
            if render_mode == "ansi":
                print(f"\nStep: {step_count} | Total Reward: {total_reward}")
            
            # Verificar si el juego terminó
            if terminated:
                print("\n" + "="*40)
                print("GAME OVER!")
                print(f"Score final: {info['score']}")
                print(f"Líneas completadas: {info['lines_cleared']}")
                print(f"Pasos totales: {step_count}")
                print("="*40)
                
                # Preguntar si quiere jugar de nuevo
                play_again = input("\n¿Jugar de nuevo? (s/n): ").strip().lower()
                if play_again == 's':
                    obs, info = env.reset()
                    total_reward = 0
                    step_count = 0
                    env.render()
                    print("\n¡Nuevo juego iniciado!")
                else:
                    break

            print(info)
                    
        except ValueError:
            print("Entrada inválida. Introduce un número del 0 al 5, o 'q' para salir.")
        except KeyboardInterrupt:
            print("\n\n¡Juego interrumpido!")
            break
    
    env.close()
    print("\n¡Gracias por jugar!")
 
 
if __name__ == "__main__":
    main()
