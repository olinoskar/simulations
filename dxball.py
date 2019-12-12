# %% DX-ball
import contextlib
with contextlib.redirect_stdout(None):
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

# New-brick probabilities
PROB_GRAY = 0.05
PROB_RED = 0.02
PROB_BLUE = 0.01



class Bricka:
    def __init__(self,course_nbr):
        pygame.init()
        self.screen=pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("dx ball(Python GameMakers)")
    
        self.clock = pygame.time.Clock()
    
        self.font=None
          
        self.init_game(course_nbr)
        
    def init_game(self,course_nbr):
        self.brownian_motion = 0
        self.brick_state = []
        self.t0 = time.time()
        self.frames_run=0
        self.lives=1
        self.score=0
        self.padVelocity = 8
        self.nbrLevelsCleared = 0
        self.state=STATE_BALL_IN_PADDLE
        self.spawn_prob = 0.5
        self.use_network = 0
        self.frame_previous_movement = -10
        self.frames_between_actions = 2  # base value
        self.network = []
        self.initial_velocity = 0  
        self.velocity_exponents = []   
        self.velocity_factor = 1  
        
        self.paddle= pygame.Rect(300,PADDLE_Y,PADDLE_WIDTH,PADDLE_HEIGHT)
        self.ball=   pygame.Rect(300,PADDLE_Y-BALL_DIAMETER,BALL_DIAMETER,BALL_DIAMETER)
        
        self.ball_vel=[self.initial_velocity,-self.initial_velocity]
        self.create_bricks(course_nbr)
  
    def create_bricks(self,course_nbr):
        matrix = 'Levels/level'+str(course_nbr)+'.csv'
        level = np.genfromtxt(matrix)            
        self.brick_state = level
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
        
        
        boost_factor = 1  # 1 => no extra effect
        
        x = [self.ball.left, self.ball.top]
        v = self.ball_vel
        
#        brickInputs = np.ravel(self.brick_state)
#        
        a = self.brick_state
        a_flat = a.reshape(a.shape[0]*a.shape[1])
        brickInputs = [int(el) for el in a_flat]       
        
        otherInputs = [x[0],x[1],v[0],v[1],self.paddle.left]

        inputs = brickInputs + otherInputs     
        outputs = Network.prop_forward(self.network, inputs)
            
        new_action_allowed = (self.frames_run - self.frame_previous_movement
                              > self.frames_between_actions)
        
        neuron_index = np.where(outputs == max(outputs))[0][0]
        
        # Network actions
        if self.use_network and new_action_allowed:
            if neuron_index == 0:  # move left
                self.paddle.left -= self.padVelocity*boost_factor
                self.frame_previous_movement = self.frames_run
                if self.paddle.left < 0:
                    self.paddle.left = 0
                    
            elif neuron_index == 1:  # stand still   
                self.paddle.left = self.paddle.left
                
            elif neuron_index == 2:  # move right
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
        iCount = -1           
            
        for brick in self.bricks:
            iCount += 1
            indices = np.where(self.brick_state == 1)
            iRow = indices[0][iCount]
            iCol = indices[1][iCount]    
            
            
            
            if self.ball.colliderect(brick):
                
                self.brick_state[iRow,iCol] = 0                
                self.score +=10
                self.ball_vel[1]=-self.ball_vel[1]
                self.bricks.remove(brick)
                
                break
            
        jCount = -1
        for brick in self.red_bricks:
            jCount += 1
            indices = np.where(self.brick_state == 2)
            iRow = indices[0][jCount]
            iCol = indices[1][jCount]   
            if self.ball.colliderect(brick):
                
                self.brick_state[iRow,iCol] = 0     
                self.score +=50

                self.ball_vel[0]=+self.ball_vel[0]*self.velocity_exponents[1]
                self.ball_vel[1]=-self.ball_vel[1]*self.velocity_exponents[1]
                self.red_bricks.remove(brick)
                
                break
            
        kCount = -1
        
        for brick in self.blue_bricks:
            kCount += 1
            indices = np.where(self.brick_state == 3)
            iRow = indices[0][kCount]
            iCol = indices[1][kCount]  
            if self.ball.colliderect(brick):
                
                self.brick_state[iRow,iCol] = 0     
                self.score +=100
                self.ball_vel[0]=+self.ball_vel[0]*self.velocity_exponents[2]
                self.ball_vel[1]=-self.ball_vel[1]*self.velocity_exponents[2]
                self.blue_bricks.remove(brick)
                
                break
            
        if len(self.bricks)==0:
            self.state=STATE_GAME_OVER
  
        if self.ball.colliderect(self.paddle):
            self.ball.top=PADDLE_Y-BALL_DIAMETER
            
            ##### Dynamic hits ##### /Gustaf
            diff = self.ball.left - self.paddle.left
            
            phi = np.pi*(1-diff/PADDLE_WIDTH)
            print(phi)

            v = 0.1 if np.random.random() < 0.5 else -0.1
            tol = 0.5
            if np.abs(phi-np.pi/2)<tol:
                phi+=v
            elif np.abs(phi-np.pi)<tol:
                phi = 3*np.pi/4
            elif np.abs(phi)<tol:
                phi = np.pi/4

                
            
            v = np.linalg.norm(self.ball_vel)
            v_new = v*self.velocity_exponents[0]
            
            self.ball_vel[0] = v_new*np.cos(phi)
            self.ball_vel[1] = -v_new*np.sin(phi) 
            
#            if np.abs(self.ball_vel[0]) < 0.1:
#                self.ball_vel[0] = 1
                
            ##### End dynamic hits #####
                
        elif self.ball.top > self.paddle.top:
            self.lives-=1
            if self.lives>0:
                self.state=STATE_BALL_IN_PADDLE
            else:
                self.state=STATE_GAME_OVER
                
    def old_brownian(self):
        r = -1 + 2*np.random.rand()
        
        v = np.linalg.norm(self.ball_vel)
        vx = self.ball_vel[0]
        vy = self.ball_vel[1]
    
        gamma = r
        beta = -vy + np.sqrt(v**2-vx**2-2*vx*gamma-gamma**2)
        
        if (np.abs(r) < 0.02 and self.ball.top < 480/3):
            print(self.ball.top)
            
            if np.sign(r) > 0:
                self.ball_vel[0] += gamma
                self.ball_vel[1] += beta
                
            if np.sign(r) < 0:
                self.ball_vel[0] -= gamma
                self.ball_vel[1] -= beta
                
    def apply_brownian_motion(self):
        r = -1 + 2*np.random.rand()
        
        if (np.abs(r) < 0.10 and self.ball.top > 480/3 and self.ball.top < 480/2):
            v = np.linalg.norm(self.ball_vel)
            
            vx = self.ball_vel[0]
            vy = self.ball_vel[1]
            
            v_new = v + self.velocity_terms[0]
            
            d_phi = (-1 + 2*np.random.rand())*np.pi/10            
            phi = np.arctan(vy/vx)
            #if vx < 0:
            #    phi = np.pi - phi
            
            self.ball_vel[0] = v_new*np.cos(phi + d_phi)
            self.ball_vel[1] = -v_new*np.sin(phi + d_phi) 
        
                

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
    
    def generate_new_bricks(self):

        indices = np.where(self.brick_state == 0)
        n = len(indices[0])
        new_vals = np.random.choice([0,1,2,3], size = n, p=[1-PROB_GRAY-PROB_BLUE-PROB_RED, PROB_GRAY, PROB_RED, PROB_BLUE])

        for i in range(n):

            x_ofs = 25 + indices[1][i]*(BRICK_WIDTH + 10)
            y_ofs = 25 + indices[0][i]*(BRICK_HEIGHT+ 5)

            if y_ofs > self.ball.top:
                continue

            self.brick_state[indices[0][i], indices[1][i]] = new_vals[i]
            
            if new_vals[i] == 1:
                self.bricks.append(pygame.Rect(x_ofs,y_ofs,BRICK_WIDTH,BRICK_HEIGHT))
            elif new_vals[i] == 2:
                self.red_bricks.append(pygame.Rect(x_ofs,y_ofs,BRICK_WIDTH,BRICK_HEIGHT))
            elif new_vals[i] == 3:
                self.blue_bricks.append(pygame.Rect(x_ofs,y_ofs,BRICK_WIDTH,BRICK_HEIGHT))


    
    def run(self,network,use_network,course_nbr,display_game,fps,max_nbr_frames,
            initial_velocity,velocity_exponents,brownian_motion,stochastic_spawning,velocity_factor):
        
        
        self.padVelocity = self.padVelocity * velocity_factor
        self.ball_vel[0] = velocity_factor*self.ball_vel[0]
        self.ball_vel[1] = velocity_factor*self.ball_vel[1]

        self.brownian_motion = brownian_motion
        self.initial_velocity = initial_velocity*velocity_factor  
        self.velocity_exponents = velocity_exponents              
        self.network = network            
        self.use_network = use_network

        while 1:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    pygame.quit()

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
                #print("game ran for ", self.frames_run, " frames")
                return self.score, self.frames_run
            self.draw_bricks()
    
            #draw paddle
            pygame.draw.rect(self.screen,BLUE,self.paddle)
    
            #draw ball
            pygame.draw.circle(self.screen,WHITE,(self.ball.left+int(BALL_RADIUS),self.ball.top + BALL_RADIUS),BALL_RADIUS)
    
            self.show_stats()
    
            if display_game:
                pygame.display.flip()

            if stochastic_spawning:
                if np.random.random() < 0.01:
                    self.generate_new_bricks()
                    self.draw_bricks()
        

    
def play_game(network,use_network=0,course_nbr=666,display_game=0,fps=50,
              max_nbr_frames=1e5, score_exponent=1, frame_exponent=1,
              initial_velocity = 7.5, velocity_exponents = [1.015, 1.025, 1.05],
              brownian_motion=0, stochastic_spawning = True, velocity_factor = 0.3):
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
    score, frames_run = b.run(network,use_network,course_nbr,
                              display_game,fps/velocity_factor,max_nbr_frames,
                              initial_velocity,velocity_exponents,
                              brownian_motion, stochastic_spawning,
                              velocity_factor)
    
    fitness = score**(np.float(score_exponent))*frames_run**(np.float(frame_exponent))
    """
    print("Fitness is defined as F = score^"+str(score_exponent) +
          "*frames_run^"+str(frame_exponent))
    print("Fitness after playing=",fitness)
    """
    return score, frames_run, fitness
