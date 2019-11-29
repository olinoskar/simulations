# %%
import sys
import pygame
import numpy as np
import time
from CreateLevel import CreateLevel
SCREEN_SIZE = 640,480

#object dimensions
BRICK_WIDTH=40
BRICK_HEIGHT=15
PADDLE_WIDTH=120
PADDLE_HEIGHT=12
BALL_DIAMETER=16
BALL_RADIUS=int(BALL_DIAMETER/2)

MAX_PADDLE_X=SCREEN_SIZE[0]-PADDLE_WIDTH
MAX_BALL_X=SCREEN_SIZE[0]-BALL_DIAMETER
MAX_BALL_Y=SCREEN_SIZE[1]-BALL_DIAMETER

#PADDLE Y CO-ORDINATE
PADDLE_Y = SCREEN_SIZE[1]-PADDLE_HEIGHT-10

#COLOR CONSTANTS
BLACK=(0,0,0)
WHITE=(255,255,255)
BLUE=(0,0,255)
BRICK_COLOR=(200,200,200)

#STATE CONSTANTS
STATE_BALL_IN_PADDLE=0
STATE_PLAYING=2
#STATE_WON=2
STATE_GAME_OVER=3

def PlayGame(network,maximumPlayTime):    
    # returns the score of the network

    class Bricka:
        def __init__(self):
            pygame.init()
            self.screen=pygame.display.set_mode(SCREEN_SIZE)
            pygame.display.set_caption("dx ball(Python GameMakers)")
        
            self.clock = pygame.time.Clock()
        
            if pygame.font:
              self.font=pygame.font.Font(None,30)
            else:
              self.font=None
        
            self.init_game()
            
        def getCurrentBrickState():  # TODO
                
            return 0
            
        def init_game(self):
            self.t0 = time.time()
            self.framesRun=0
            self.lives=1
            self.score=0
            self.padVelocity = 12
            self.nbrLevelsCleared = 0
            
            self.state=STATE_BALL_IN_PADDLE
    
            self.paddle= pygame.Rect(300,PADDLE_Y,PADDLE_WIDTH,PADDLE_HEIGHT)
            self.ball=   pygame.Rect(300,PADDLE_Y-BALL_DIAMETER,BALL_DIAMETER,BALL_DIAMETER)
            
            
            v = 7
            self.ball_vel=[v,-v]
            self.create_bricks()
      
        def create_bricks(self):
            self.ball_vel[0] += self.ball_vel[0]/2
            self.ball_vel[1] += self.ball_vel[0]/2
            print(self.ball_vel)
            const = 0.1*self.nbrLevelsCleared
            CreateLevel(const)
            matrix = 'Levels/level.csv'
            level = np.genfromtxt(matrix)   
            level_width = len(level[0])
            level_height = len(level)
            y_ofs=25
            self.bricks=[]
            for i in range(level_height):
                x_ofs=25
                for j in range(level_width):
                    if level[i,j] != 0:
                        self.bricks.append(pygame.Rect(x_ofs,y_ofs,BRICK_WIDTH,BRICK_HEIGHT))
                    x_ofs += BRICK_WIDTH + 10
                y_ofs += BRICK_HEIGHT+5
        def draw_bricks(self):
            for brick in self.bricks:
                pygame.draw.rect(self.screen,BRICK_COLOR,brick)
    
        def check_input(self):  # Edit this to take input from neural network
            
            # Check time passed
            t1 = time.time()
            if t1 - self.t0 > maximumPlayTime:
                return score
                pygame.quit()
                exit()
            
            
            keys=pygame.key.get_pressed()
            
            # FEED FOWARD HERE AND RETURN output
            output = -1+2*np.random.rand()
            self.framesRun += 1
            
            if np.remainder(self.framesRun,1) == 0:
            
                boostFactor = 10
                
                if output < 0:
                    self.paddle.left-= self.padVelocity*boostFactor
                    if self.paddle.left < 0:
                        self.paddle.left = 0
                elif output > 0:
                    self.paddle.left += self.padVelocity*boostFactor
                    if self.paddle.left > MAX_PADDLE_X:
                        self.paddle.left = MAX_PADDLE_X
      
            # Below: possible to assign own inputs as well, so keep these for the 
            # time being.
            if keys[pygame.K_LEFT]:
                self.paddle.left-= self.padVelocity
                if self.paddle.left < 0:
                    self.paddle.left = 0
      
            if keys[pygame.K_RIGHT]:
                self.paddle.left += self.padVelocity
                if self.paddle.left > MAX_PADDLE_X:
                    self.paddle.left = MAX_PADDLE_X
      
            if keys[pygame.K_SPACE] and self.state== STATE_BALL_IN_PADDLE:
                self.ball_vel=[5,-5]
                self.state=STATE_PLAYING
            elif keys[pygame.K_RETURN] and (self.state==STATE_GAME_OVER):
                self.init_game()
    
        def move_ball(self):
            self.ball.left += self.ball_vel[0]
            self.ball.top  += self.ball_vel[1]
      
            if self.ball.left<=0:
                self.ball.left=0
                self.ball_vel[0]=-self.ball_vel[0]
            elif self.ball.left>= MAX_BALL_X:
                self.ball.left =MAX_BALL_X
                self.ball_vel[0]=-self.ball_vel[0]
        
            if self.ball.top < 0:
                self.ball.top=0
                self.ball_vel[1]=-self.ball_vel[1]
    
        def handle_collisions(self):
            for brick in self.bricks:
                if self.ball.colliderect(brick):
                    self.score +=3
                    self.ball_vel[1]=-self.ball_vel[1]
                    self.bricks.remove(brick)
                    break
            if len(self.bricks)<5:
                self.nbrLevelsCleared += 1
                self.create_bricks()
      
            if self.ball.colliderect(self.paddle):
                self.ball.top=PADDLE_Y-BALL_DIAMETER
                self.ball_vel[1]=-self.ball_vel[1]
            elif self.ball.top > self.paddle.top:
                self.lives-=1
                if self.lives>0:
                    self.state=STATE_BALL_IN_PADDLE
                else:
                    self.state=STATE_GAME_OVER
    
        def show_stats(self):
            if self.font:
                font_surface=self.font.render("SCORE:" + str(self.score)+"LIVES:"+str(self.lives),False,WHITE)
                self.screen.blit(font_surface,(205,5))
    
        def show_message(self,message):
            if self.font:
                size=self.font.size(message)
                font_surface=self.font.render(message,False,WHITE)
                x= (SCREEN_SIZE[0]-size[0])/2
                y=(SCREEN_SIZE[1]-size[1])/2
                self.screen.blit(font_surface,(x,y))
        
        def run(self):
            while 1:
                for event in pygame.event.get():
                    if event.type==pygame.QUIT:
                        pygame.quit()
                        exit()
                    
                self.clock.tick(50)
                self.screen.fill(BLACK)
                self.check_input()
        
                if self.state==STATE_PLAYING:
                    self.move_ball()
                    self.handle_collisions()
                elif self.state==STATE_BALL_IN_PADDLE:
                    self.ball.left=self.paddle.left+ self.paddle.width/2
                    self.ball.top=self.paddle.top-self.ball.height
                    self.show_message("Press space to launch the ball")
                elif self.state==STATE_GAME_OVER:
                    self.show_message("Game over.press enter to play again")
                #elif self.state==STATE_WON:
                #    self.show_message("You won! press enter to play again")
        
                self.draw_bricks()
        
                #draw paddle
                pygame.draw.rect(self.screen,BLUE,self.paddle)
        
                #draw ball
                pygame.draw.circle(self.screen,WHITE,(self.ball.left+int(BALL_RADIUS),self.ball.top + BALL_RADIUS),BALL_RADIUS)
        
                self.show_stats()
        
                pygame.display.flip()
        
    if __name__== "__main__":
        Bricka().run()

return self.score

network = []
maximumPlayTime = 10  # seconds
score = PlayGame(network,maximumPlayTime)