# %% DX-ball
import pygame
import numpy as np
import time
from Network import Network

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
RED = (255,0,0)

#STATE CONSTANTS
STATE_BALL_IN_PADDLE=0
STATE_PLAYING=2
STATE_GAME_OVER=3


class Bricka:
    def __init__(self,course_nbr):
        pygame.init()
        self.screen=pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("dx ball(Python GameMakers)")
    
        self.clock = pygame.time.Clock()
    
        self.font=None
          
        self.init_game(course_nbr)
        
    def init_game(self,course_nbr):
        self.t0 = time.time()
        self.frames_run=0
        self.lives=1
        self.score=0
        self.padVelocity = 18
        self.nbrLevelsCleared = 0
        self.state=STATE_BALL_IN_PADDLE
        self.spawn_prob = 0.5
        self.use_network = 0
        self.frame_previous_movement = -10
        self.frames_between_actions = 2  # base value
        self.network = []
        
        self.crate= []
        #pygame.Rect(self.ball.left,self.ball.top,20,20)
        self.paddle= pygame.Rect(300,PADDLE_Y,PADDLE_WIDTH,PADDLE_HEIGHT)
        self.ball=   pygame.Rect(300,PADDLE_Y-BALL_DIAMETER,BALL_DIAMETER,BALL_DIAMETER)
        
        
        v = 5
        self.ball_vel=[v,-v]
        self.create_bricks(course_nbr)
  
    def create_bricks(self,course_nbr):
        self.ball_vel[0] += self.ball_vel[0]/2
        self.ball_vel[1] += self.ball_vel[0]/2
        matrix = 'Levels/level'+str(course_nbr)+'.csv'
        level = np.genfromtxt(matrix)   
        level_width = len(level[0])
        level_height = len(level)
        y_ofs=25
        self.bricks=[]
        self.red_bricks=[]
        self.blue_bricks=[]
        for i in range(level_height):
            x_ofs=25
            for j in range(level_width):
                if level[i,j] == 1:
                    self.bricks.append(pygame.Rect(x_ofs,y_ofs,BRICK_WIDTH,BRICK_HEIGHT))
                elif level[i,j] == 2:
                    self.red_bricks.append(pygame.Rect(x_ofs,y_ofs,BRICK_WIDTH,BRICK_HEIGHT))
                elif level[i,j] == 3:
                    self.blue_bricks.append(pygame.Rect(x_ofs,y_ofs,BRICK_WIDTH,BRICK_HEIGHT))
                x_ofs += BRICK_WIDTH + 10
            y_ofs += BRICK_HEIGHT+5
            
    def draw_bricks(self):
        for brick in self.bricks:
            pygame.draw.rect(self.screen,BRICK_COLOR,brick)
        for brick in self.red_bricks:
            pygame.draw.rect(self.screen,RED,brick)
        for brick in self.blue_bricks:
            pygame.draw.rect(self.screen,(0,0,255),brick)
    
    def check_input(self,course_nbr):  # Edit this to take input from neural network
        keys=pygame.key.get_pressed()
        self.frames_run += 1
        
        boost_factor = 1  # 1 => no extra effect
        
        x = [self.ball.left, self.ball.top]
        v = self.ball_vel
        inputs = [x[0],x[1],v[0],v[1]]
             
        output = Network.prop_forward(self.network, inputs)
            
        new_action_allowed = (self.frames_run - self.frame_previous_movement
                              > self.frames_between_actions)

        
        # Network actions
        if self.use_network and new_action_allowed:
            if output < 0.33:
                self.paddle.left-= self.padVelocity*boost_factor
                self.frame_previous_movement = self.frames_run
                if self.paddle.left < 0:
                    self.paddle.left = 0
                    
            # if output \in [0.33, 0.66]: <stand still>
                    
            elif output > 0.66:
                self.paddle.left += self.padVelocity*boost_factor
                self.frame_previous_movement = self.frames_run
                if self.paddle.left > MAX_PADDLE_X:
                    self.paddle.left = MAX_PADDLE_X
  
        # Below: possible to assign own inputs as well, so keep these for the 
        # time being.
        if keys[pygame.K_LEFT]:
            self.paddle.left -= self.padVelocity
            if self.paddle.left < 0:
                self.paddle.left = 0
  
        if keys[pygame.K_RIGHT]:
            self.paddle.left += self.padVelocity
            if self.paddle.left > MAX_PADDLE_X:
                self.paddle.left = MAX_PADDLE_X
  
        if self.state== STATE_BALL_IN_PADDLE:
            self.ball_vel=[5,-5]  # always starts ball in the same 
            self.state=STATE_PLAYING
        elif keys[pygame.K_RETURN] and (self.state==STATE_GAME_OVER):
            self.init_game(course_nbr)

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
                self.score +=10
                self.ball_vel[1]=-self.ball_vel[1]
                self.bricks.remove(brick)
                
                break
                    
        for brick in self.red_bricks:
            if self.ball.colliderect(brick):
                self.score +=50
                self.ball_vel[1]=-1.025*self.ball_vel[1]
                self.red_bricks.remove(brick)
                
                break
            
        for brick in self.blue_bricks:
            if self.ball.colliderect(brick):
                self.score +=100
                self.ball_vel[1]=-1.10*self.ball_vel[1]
                self.blue_bricks.remove(brick)
                
                break
            
        if len(self.bricks)==0:
            self.state=STATE_GAME_OVER
  
        if self.ball.colliderect(self.paddle):
            self.ball.top=PADDLE_Y-BALL_DIAMETER
            
            ##### Dynamic hits ##### /Gustaf
            diff = self.ball.left - self.paddle.left
            
            phi = np.pi*(1-diff/PADDLE_WIDTH)
            
            v = np.linalg.norm(self.ball_vel)
            
            self.ball_vel[0] = v*np.cos(phi)
            self.ball_vel[1] = -1.015*self.ball_vel[1]  # increasing y-speed with each hit
                
            q = -1 + 2*np.random.rand()
            s = np.sign(q)
            
            if self.ball_vel[0] == 0:
                self.ball_vel[0] = -s
            
            if np.abs(self.ball_vel[0]) < 0.1:
                print("xv = 0")
                self.ball_vel[0] = 10
            
            ##### End dynamic hits #####
                
        elif self.ball.top > self.paddle.top:
            self.lives-=1
            if self.lives>0:
                self.state=STATE_BALL_IN_PADDLE
            else:
                self.state=STATE_GAME_OVER

    def show_stats(self):
        if self.font:
            font_surface=self.font.render("Score:" + str(self.score)+" Time:"+str(np.floor(self.frames_run/10)),False,WHITE)
            self.screen.blit(font_surface,(205,5))

    def show_message(self,message):
        if self.font:
            size=self.font.size(message)
            font_surface=self.font.render(message,False,WHITE)
            x= (SCREEN_SIZE[0]-size[0])/2
            y=(SCREEN_SIZE[1]-size[1])/2
            self.screen.blit(font_surface,(x,y))
            
    
    def run(self,network,use_network,course_nbr,display_game,fps,max_nbr_frames):
        
        while 1:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    pygame.quit()
                    
            self.network = network            
            self.use_network = use_network
            self.clock.tick(fps)
            self.screen.fill(BLACK)
            self.check_input(course_nbr)
            
            if self.frames_run > max_nbr_frames:
                self.state = STATE_GAME_OVER
    
            if self.state==STATE_PLAYING:
                self.move_ball()
                self.handle_collisions()
            elif self.state==STATE_BALL_IN_PADDLE:
                self.ball.left=self.paddle.left+ self.paddle.width/2
                self.ball.top=self.paddle.top-self.ball.height
                
            elif self.state==STATE_GAME_OVER:
                print("game ran for ", self.frames_run, " frames")
                return self.score, self.frames_run
            self.draw_bricks()
    
            #draw paddle
            pygame.draw.rect(self.screen,BLUE,self.paddle)
    
            #draw ball
            pygame.draw.circle(self.screen,WHITE,(self.ball.left+int(BALL_RADIUS),self.ball.top + BALL_RADIUS),BALL_RADIUS)
    
            self.show_stats()
    
            if display_game:
                pygame.display.flip()
        

    
def play_game(network,use_network=1,course_nbr=666,display_game=0,fps=50,max_nbr_frames=1000,
              score_exponent=1, frame_exponent=1):
    '''
    [] Courses are defined as 'level<course_nbr>.csv'. The standard course is 666.
    
    [] Fitness measure is defined as  F = score^a*frames_run^b, where 
        a = score_exponent
        b = frame_exponent
    
    params:
        * Network network: Network object that plays the game
        * use_network (0 or 1): if 1, only user inputs change pad position,
            otherwise, the network plays on its own.
        * int display_game (0 or 1): show game visuals or not 
        * int fps: essentially game speed
        * float score_exponent
        * float frame_exponent        
    
    returns: 
        * int score: from destroying bricks
        * int frames_run: number of frames played before termination
    '''
        
    ''' TODO:
            * let user input: int frames_between_actions
    '''
    b = Bricka(course_nbr)
    score, frames_run = b.run(network,use_network,course_nbr,display_game,fps,max_nbr_frames)
    
    fitness = score**(np.float(score_exponent))*frames_run**(np.float(frame_exponent))
    
    print("Fitness is defined as F = score^"+str(score_exponent) +
          "*frames_run^"+str(frame_exponent))
    print("Fitness after playing=",fitness)
    
    return score, frames_run, fitness
