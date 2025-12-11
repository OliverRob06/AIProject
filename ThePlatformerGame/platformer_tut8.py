import pygame 
import numpy as np 
from pygame.locals import *
import math
import sys
	
# initialize pygame
pygame.init()
pygame.font.init()	
# The Environment Global Variables
scale = 1
slowerEnemy = False
clock = pygame.time.Clock()
fps = 120

screen_width = 1000*scale
screen_height = 1000*scale

render = False
actionOverXFrames = 5
show_hitboxes = False
timeLimit = 10

# create window
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Platformer')

# define game variables
tile_size = 50*scale
main_menu = True
score = 0

# define font variables
font_score = pygame.font.SysFont('Bauhaus 93', 30)

# define colour
black = (0, 0,0)

#load images
sun_img = pygame.image.load('./ThePlatformerGame/img/sun.png')
bg_img = pygame.image.load('./ThePlatformerGame/img/sky.png')
win_img = pygame.image.load('./ThePlatformerGame/img/youwin.png')
# world data 2d array
# dirt block = 1, grass block = 2, enemy = 3, lava = 6, coin = 7, goal = 8
world_data = [
[1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 8, 0, 0, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 0, 2, 2, 0, 7, 1], 
[1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 
[1, 0, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 7, 2, 2, 1, 1], 
[1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 2, 6, 6, 2, 2, 1, 1, 1, 1], 
[1, 0, 0, 2, 2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 
[1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 1, 0, 7, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 1], 
[1, 1, 2, 2, 0, 0, 3, 0, 0, 2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 1], 
[1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 7, 1], 
[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 2, 1], 
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 2, 2, 2, 1, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1], 
[1, 0, 0, 0, 7, 0, 0, 0, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 0, 0, 0, 2, 2, 6, 6, 1, 1, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]
# jed comment on what the function please 
def putInRange(number, minV, maxV):
    # less than min
    output = max(number, minV)
    # more than max
    output = min(output, maxV)
    return output

# platformer Environment Class 
class platformerEnv:
	# Initialize the environment
	def __init__(self):

		#Initialize Environment with default attributes
		self.player = Player()

		#create dummy coin for showing the score
		self.score_coin = Coin(tile_size // 2, tile_size // 2)
		self.world = World(world_data)
		self.world.coin_group.add(self.score_coin)

		# set world data and tile size
		self.world_data = world_data
		self.tile_size = tile_size

		# initialize timing variables
		self.start_time = pygame.time.get_ticks()
		self.gameWon = False
		self.wonTime = 0	
		self.frame = 0
		self.game_over = 0

		# initialize previous distance to goal
		self.prevDistance = 1
		# initialize last x position
		self.lastX = self.player.rect.x
	
	# Function to draw text on screen
	def draw_text(self, text, font, text_col, x, y):
		# error handling for font not initialized
		if not pygame.font.get_init():
			pygame.font.init()
		# render the text
		img = font.render(text, True, text_col)
		# update screen with text
		screen.blit(img, (x, y))

	# Reset the level	
	def reset_level(self):
		# Reset player and clear groups
		self.player.reset()
		self.world.blob_group.empty()
		self.world.coin_group.empty()
		self.world.lava_group.empty()
		self.world.exit_group.empty()

		# Create new world
		world = World(world_data)
		self.world_data = world_data
		self.tile_size = tile_size

		#create dummy coin for showing the score
		score_coin = Coin(tile_size // 2, tile_size // 2)
		world.coin_group.add(score_coin)
		return world

	# Prints values for debugging
	def d14Vector(self):
		# player position
		playerX = self.player.rect.x
		playerY = self.player.rect.y

		#player input
		key = pygame.key.get_pressed()
		left = key[pygame.K_LEFT]
		right = key[pygame.K_RIGHT]
		space = key[pygame.K_SPACE]
		
		#nothing pressed
		if not left and not right and not space:
			nothing = True
		else:
			nothing = False

		# player in air
		playerInAir = self.player.in_air 

		# player vertical velocity
		playerVelY = self.player.rect.y - (self.player.rect.y - self.player.vel_y)

		# see terrain infront of player
		Height = self.player.getHeight()

		# enemy positions
		for enemy in self.world.blob_group:
			enemyX = enemy.rect.x
			enemyY = enemy.rect.y

		# goal position
		playX = self.player.rect.x
		playY = self.player.rect.y
		for exit in self.world.exit_group:
			exitX = exit.rect.x
			exitY = exit.rect.y

		# distances
		closestEnemyDistance = self.getClosestEnemyDistance(self.player.rect.x, self.player.rect.y)
		target = self.closest_sprite(self.player, self.world.coin_group) if self.world.coin_group else self.closest_sprite(self.player, self.world.exit_group)

		# get the pixel looking at
		if self.player.direction == -1:  #facing left
			pixelsToCheckx = self.player.rect.x-1 #block on left
			pixelsToCheckY = self.player.rect.y+79
		else:
			pixelsToCheckx = self.player.rect.x+50#block on right
			pixelsToCheckY = self.player.rect.y+79
		#print("Pixels to check X: ", pixelsToCheckx)
		#print("Pixels to check Y: ", pixelsToCheckY)
		

		# time spent
		timeSpent = pygame.time.get_ticks() - self.start_time
		
		# prints
		"""print("Player X: ", playerX)
		print("Player Y: ", playerY)

		print("key LEFT: ", left)
		print("key RIGHT: ", right)
		print("key SPACE: ", space)
		print("No Input: ", nothing)

		print("Player In Air: ", playerInAir)
		print("Player Vel Y: ", playerVelY)
		print("Height: ", Height)

		print("Enemy X: ", enemyX)
		print("Enemy Y: ", enemyY)

		print("Closest Enemy Distance: ", closestEnemyDistance)
		print("Closest Goal/Coin Distance: ", closestGoalOrCoin)

		print("Goal x: ", exitX)
		print("Goal y: ", exitY)
		
		print("Player x: ", playX)
		print("Player y: ", playY)

		print("Pixels to check X: ", pixelsToCheckx)
		print("Pixels to check Y: ", pixelsToCheckY)


		print("Time Spent (ms): ", timeSpent)"""

	# Get distance to closest enemy
	def getClosestEnemyDistance(self, playerX, playerY):
		# Player position
		playerX = self.player.rect.centerx
		playerY = self.player.rect.y+55
		decidedEnemyX = 0

		# Initialize minimum distance and closest enemy position
		minDistance = float('inf')
		
		# For each enemy in the blob group
		for enemy in self.world.blob_group:
			# Enemy position
			enemyX = enemy.rect.centerx
			enemyY = enemy.rect.y+12
			#Slime height = 52, player height = 80

			# Calculate distance to player using pythagorean theorem
			distance = ((enemyX - playerX) ** 2 + ((enemyY) - (playerY)) ** 2) ** 0.5
			# If this distance is less than the minimum distance found so far
			if distance < minDistance:
				# Update minimum distance and closest enemy position
				minDistance = distance
				decidedEnemyX = enemyX

		return decidedEnemyX, minDistance
	
	# Get distance to closest goal or coin
	def getClosestGoalOrCoinDistance(self, playerX, playerY):
		minDistance = float('inf')
		decidedCoinX = 0
		for coin in self.world.coin_group:
			coinX = coin.rect.x
			coinY = coin.rect.y
			distance = ((coinX - playerX - 40) ** 2 + ((coinY+52) - (playerY+80)) ** 2) ** 0.5
			if distance < minDistance:
				minDistance = distance
				decidedCoinX = coinX
		
		for goal in self.world.exit_group:
			goalX = goal.rect.x
			goalY = goal.rect.y
			distance = ((goalX + 35 - playerX) ** 2 + ((goalY+ - playerY)) ** 2) ** 0.5
			if distance < minDistance:
				minDistance = distance
				decidedCoinX = goalX

		return decidedCoinX, minDistance
	# get the disance to the closest sprite in a group
	def closest_sprite(self, player, sprites):
		# initialize minimum distance to a large value
		min_dist = float('inf')
		# initialize closest sprite to None
		closest = None

		#for each sprite in the group
		for s in sprites:
			# calculate the distance to the player, favouring the x-axis
			dist = math.hypot(
				player.rect.centerx - s.rect.x,
				(player.rect.centery - s.rect.y)*10,
			)
			# if this distance is less than the minimum distance found so far
			if dist < min_dist:
				# update minimum distance and closest sprite
				min_dist = dist
				closest = s

		# return the closest sprite found
		return closest

	## Reset, Step and getState all are in a class with player, world, enemies as elements
	# Used for the AI resetting
	def reset(self):
		global score
		self.player.reset()
		self.game_over = 0
		score = 0
		# initialize previous distance to goal
		self.prevDistance = 1

		# initialize world
		self.world = self.reset_level()
		# initialize last x position
		self.lastX = self.player.rect.x

		# initialize timing variables
		self.start_time = pygame.time.get_ticks()
		
		# Return the initial state of the game and an empty info dict
		return self.get_state(), {}

	
		# AI takes action and returns new state and reward associated
	
	def step(self, action):
		global render
		global score
		global actionOverXFrames
		global timeLimit

		# For calculating reward
		# Starts at -0.5 to incentivise taking less time
		#reward = -0.5*actionOverXFrames
		reward = 0
		wasInAir = self.player.in_air
		for x in range(actionOverXFrames):

			# Translate the given number corresponding to an action, and make subsequent movement.
			self.game_over = self.player.update(action, self.world, self.game_over)
			
			# If Timeout Kill Player
			timePassed = (pygame.time.get_ticks() - self.start_time) / 1000
			if timePassed > timeLimit:
				self.game_over = -1


			isDoingNothing = action == 4
			isMiddair = self.player.in_air

			isFacingLeft = self.player.direction == -1 
			isFacingRight = self.player.direction == 1 

			isWalkingLeft = action == 0
			isJumpingLeft = action == 1
			isWalkingRight = action == 2
			isJumpingRight = action == 3

			isJumping = action == 1 or action == 3

			curBlock = self.player.getHeight() 

			isFacingWall = curBlock == 3
			isFacing2BlockHigh = curBlock == 2
			isFacing1BlockHigh = curBlock == 1
			isFacingNothing = curBlock == -1
			isFacing1BlockDrop = curBlock == -2
			isFacingDropOrLava = curBlock == -3
			
			if isFacingLeft:

				if isFacingWall:

					if isWalkingLeft or isJumpingLeft:
						reward-=20
					
					elif isWalkingRight or isJumpingRight:
						reward+=8

				elif isFacing1BlockDrop or isFacingDropOrLava or isFacing1BlockHigh or isFacing2BlockHigh:
					if isMiddair:
							pass
					else:
						if isJumpingLeft:
							# Jump over
							reward+=10
						elif isWalkingLeft:
							# Waste Movement
							reward-=20
						elif isWalkingRight or isJumpingRight:
							# Retreat/Back Up
							pass

				elif isFacingNothing:

					if isWalkingLeft:
						reward+=5
					else:
						reward-=5

				

			elif isFacingRight:

				if isFacingWall:

					if isWalkingLeft or isJumpingLeft:
						reward+=5
					
					elif isWalkingRight or isJumpingRight:
						reward-=20

				elif isFacing1BlockDrop or isFacingDropOrLava or isFacing1BlockHigh or isFacing2BlockHigh:
					if isMiddair:
							pass
					else:
						if isJumpingRight:
							# Jump over
							reward+=10
						elif isWalkingRight:
							# Waste Movement
							reward-=20
						elif isWalkingLeft or isJumpingLeft:
							# Retreat/Back Up
							pass

				elif isFacingNothing:

					if isWalkingRight:
						reward+=5
					else:
						reward-=5
			
			coinDirection, coinDistance = self.getClosestGoalOrCoinDistance(self.player.rect.x, self.player.rect.y)
			# incentivise getting closer to goal/coin
			if self.prevDistance > coinDistance:
				reward += 0.2
				#print ("rewarded +2 for going to coin")
			else:
				reward -= 0.2
				#print ("punished -1 for going away from coin")
			# If player has died
			if self.game_over == -1:
				reward-=1000
				# print ("Died - 1000")
			# If player reaches wins
			if self.game_over == 1:
				reward+=1000
			#print (self.player.getHeight())
			# If player reaches coin (checkpoint)
			if pygame.sprite.spritecollide(self.player, self.world.coin_group, True):
				# update on screen score
				score += 1
				# Update Agent reward
				reward += 50
				# print ("Got COin +700")

			
			
			"""
			# Deincentivise looking at a wall
			# If looking at a wall and facing left
			if self.player.getHeight() == 3 and self.player.direction == -1:
				# If action is going towards the wall
				if action in [0,1]:
					reward-=60
					print ("punished-60 for wall going into")
				else:
					reward+=40
					print ("Rewarded+40 for wall going away")

			# If looking at a wall and facing right
			elif self.player.getHeight() == 3 and self.player.direction == 1:
				# If action is going towards the wall
				if action in [2,3]:
					reward-=60
					print ("punished-60 for wall going into")
				else:
					reward+=40
					print ("Rewarded+40 for wall going away")


			# if jumping
			if action in [1, 3]:
				# And on ground
				if self.player.in_air == False:
					currentRelativeHeight = self.player.getHeight()

					# reward jumping over a gap or small blocks
					if currentRelativeHeight in [1, 2, -2, -3]: 
						reward += 50.0
						print ("Rewarded+50 for jumping over block/gap")

					else:
						reward -= 60.0 
						print ("Rewarded-60 for jumping randomly")

			if self.player.getHeight() == 1:
                
                # Walking into left block 
				if self.player.direction == -1:
					if action == 0: # 0 = Walk Left (bad), 1 = Jump Left (good)
						reward -= 20.0 
						print("Bonk! Penalty for walking LEFT into block")

                # CWalk into right block
				elif self.player.direction == 1:
					if action == 2: 
						reward -= 20.0 
						print("Bonk! Penalty for walking RIGHT into block")
			"""
			"""There are six discrete actions available: 
		0: go left
		1: go left and up
		2: go right
		3: go right and up
		4: do nothing
		"""
			
			"""
			# Decentivise running into lava
			# When faced with lava
			if self.player.getHeight() == -3:
				isRushingLeft = (self.player.direction == -1 and action == 0)
				isRushingRight = (self.player.direction == 1 and action == 2)

				if isRushingLeft or isRushingRight:
					reward -= 100.0
					print ("Rushing -100")
				else:
					print ("didnt rush into lava, self.player.direction", self.player.direction , "action ", action)

			"""
			
			
			# Check game is over (Win or Lose)
			terminated = False
			if (self.game_over == -1 or self.game_over == 1):
				terminated = True

			if render:
				# Update Screen
				screen.blit(bg_img, (0, 0))
				screen.blit(sun_img, (100, 100))
				self.world.draw()

			# draw player
			screen.blit(self.player.image, self.player.rect)

			# Handle Pygame events (keep window from freezing)
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					sys.exit()

			# Calculate time survived
			if self.gameWon >= 1:
				current_time = self.wonTime
			else:
				current_time = pygame.time.get_ticks() - self.start_time

				# Update timing variables and screen
				seconds = current_time // 1000
				milliseconds = current_time % 1000
				timer_text = f"Time: {seconds}.{milliseconds:03d}"
				if render:
					self.draw_text(' X ' + str(score), font_score, black, ((tile_size-5)*scale), (8*scale))
					self.draw_text(timer_text, font_score, black, (screen_width-200)*scale, 4*scale)
				# Move enemies
				self.world.blob_group.update()
				
				if render:
					# Update sprites
					self.world.blob_group.draw(screen)
					self.world.lava_group.draw(screen)
					self.world.coin_group.draw(screen)
					self.world.exit_group.draw(screen)

				# update tick
				clock.tick(fps)
				self.frame += 1
				if render:
					pygame.display.update()

				# update prev distance to goal
				coinDirection, self.prevDistance = self.getClosestGoalOrCoinDistance(self.player.rect.x, self.player.rect.y)

				if terminated:
					break
				# stop movement in 4 frames if landing or start falling
				if wasInAir != self.player.in_air:
					break
				wasInAir = self.player.in_air
				if self.game_over == -1:
					break
			
		# Return new observation given new state, reward calculated and game over
		return self.get_state(), reward, terminated, {}
			
			#if platformE.frame % 20 == 0:
			#	platformE.d14Vector()

	# Returns observation of current state
	def get_state(self):
		# create empty set for observation
		state_vector = []

		# Adding values to set that will become tensor/observation

		# Adds player x position 
		playerx = round(self.player.rect.x /(tile_size * 20),3)
		state_vector.append(playerx)
		

		# Adds player y position 
		playery = round(self.player.rect.y /(tile_size * 20),3)
		state_vector.append(playery)
		
		# Adds player in air boolean as 1 or 0 
		state_vector.append(int(self.player.in_air))
		
		
		# Adds player vertical velocity 
		verticalV = round((self.player.rect.y - (self.player.rect.y - self.player.vel_y) / 15)/1000,3)
		state_vector.append(verticalV)
		

		# Adds player horizontal velocity 
		horV = self.player.dx / 5.0  
		state_vector.append(horV)
		

		# Adds player direction
		state_vector.append(self.player.direction)
		

		# Add terrain infront of players relative height to player
		relH = round(self.player.getHeight()/3,3)
		state_vector.append(relH)		
		
		
		# Add distance to nearest enemy
		enemyX, enemyDistance = self.getClosestEnemyDistance(self.player.rect.x, self.player.rect.y)
		enemyDistance = putInRange(enemyDistance,-300,300)
		enemyDistance = round(enemyDistance/300,3)
		state_vector.append(enemyDistance)
		
		# Enemy Direction
		if self.player.rect.x>enemyX:
			state_vector.append(0)
		else:
			state_vector.append(1)

		# Add distance to nearest coin or goal
		coinX, goalCoinDistance = self.getClosestGoalOrCoinDistance(self.player.rect.x, self.player.rect.y)
		goalCoinDistance = putInRange(goalCoinDistance,-300,300)
		goalCoinDistance = round(goalCoinDistance/300,3)
		state_vector.append(goalCoinDistance)

		# Coin Direction
		if self.player.rect.x>coinX:
			state_vector.append(0)
		else:
			state_vector.append(1)
		
		# Display Values
		if False:
			print("playerX: ", playerx)
			print("playery: ", playery)
			print("int(self.player.in_air): ", int(self.player.in_air))
			print("verticalV: ", int(verticalV))
			print("horizontalV: ", int(horV))
			print("self.player.direction: ", self.player.direction)
			print("relH: ", relH)
			print("enemyDistance: ", enemyDistance)
			print("goalCoinDistance: ", goalCoinDistance)
		# Returns Set turned into NumPy Float Tensor 
		return np.array(state_vector, dtype=np.float32)

# Player class
class Player():
	# Initialize player
	def __init__(self):
		self.reset()

	# Fucntion to update player position (moving the player)
	def update(self, action, world, game_over):
		global screen
		# action corresponds to either 10 for player input, or 0-5 for ai input
		self.dx = 0
		self.dy = 0
		self.walk_cooldown = 5
		"""
		There are six discrete actions available: 
		0: go left
		1: go left and up
		2: go right
		3: go right and up
		4: do nothing
		"""
		if game_over == 0:
			# Use AI actions for inputs
			if (action<6):
				# AI jumping with varients (up left, up right)
				if (action==1 or action==3) and self.jumped == False and self.in_air == False:
					self.vel_y = -15
					self.jumped = True
				# Reset jump when not jumping
				if (action==1 or action==3 or action==4) == False:
					self.jumped = False
				
				# AI left movements with varients (left, left up)
				if (action==0 or action==1):
					self.dx -= 5*scale
					self.counter += 1
					self.direction = -1
				
				# AI right movements with varients (right, right up)
				if (action==2 or action==3):
					self.dx += 5*scale
					self.counter += 1
					self.direction = 1
				
				# No left/right movement input from AI
				if (action==0 or action==1) == False and (action==2 or action==3) == False:
					self.counter = 0
					self.index = 0
					if self.direction == 1:
						self.image = self.images_right[self.index]
					if self.direction == -1:
						self.image = self.images_left[self.index]

			# Use player keyboard inputs
			else:
				# Human jumping
				key = pygame.key.get_pressed()
				if key[pygame.K_SPACE] and self.jumped == False and self.in_air == False:
					self.vel_y = -15
					self.jumped = True

				# Reset jump when not jumping
				if key[pygame.K_SPACE] == False:
					self.jumped = False
				
				# Human left movement
				if key[pygame.K_LEFT]:
					self.dx -= 5*scale
					self.counter += 1
					self.direction = -1
				
				# Human right movement
				if key[pygame.K_RIGHT]:
					self.dx += 5*scale
					self.counter += 1
					self.direction = 1
				
				# No left/right movement input from human
				if key[pygame.K_LEFT] == False and key[pygame.K_RIGHT] == False:
					self.counter = 0
					self.index = 0
					if self.direction == 1:
						self.image = self.images_right[self.index]
					if self.direction == -1:
						self.image = self.images_left[self.index]

			#handle animation
			if self.counter > self.walk_cooldown:
				self.counter = 0	
				self.index += 1
				if self.index >= len(self.images_right):
					self.index = 0
				
				if self.direction == 1:
					self.image = self.images_right[self.index]
				
				if self.direction == -1:
					self.image = self.images_left[self.index]


			#add gravity
			self.vel_y += 1
			if self.vel_y > 10 :
				self.vel_y = 10
			self.dy += self.vel_y

			#check for collision
			self.in_air = True
			for tile in world.tile_list:
				#check for collision in x direction
				if tile[1].colliderect(self.rect.x + self.dx, self.rect.y, self.width, self.height):
					self.dx = 0
				#check for collision in y direction
				if tile[1].colliderect(self.rect.x, self.rect.y + self.dy, self.width, self.height):
					#check if below the ground i.e. jumping
					if self.vel_y < 0:
						self.dy = tile[1].bottom - self.rect.top
						self.vel_y = 0
					#check if above the ground i.e. falling
					elif self.vel_y >= 0:
						self.dy = tile[1].top - self.rect.bottom
						self.vel_y = 0
						self.in_air = False

			#check for collision with enemies
			if pygame.sprite.spritecollide(self, world.blob_group, False):
				game_over = -1

			#check for collision with lava
			if pygame.sprite.spritecollide(self, world.lava_group, False):
				game_over = -1
			
			#check for collision with exit
			if pygame.sprite.spritecollide(self, world.exit_group, False):
				game_over = 1
			
			#update player coordinates
			self.rect.x += self.dx
			self.rect.y += self.dy

		# draw ghost to screen upon death	
		elif game_over == -1:
			self.image = self.dead_image
			if self.rect.y > 200:
				self.rect.y -= 5

		#draw player onto screen
		screen.blit(self.image, self.rect)
		#shows hitboxes
		if show_hitboxes:
			pygame.draw.rect(screen, (255, 255, 255), self.rect, 2)
		
		return game_over
	
	def getHeight(self):
		# Player position
		xPos = self.rect.x
		yPos = self.rect.y

		# Pixels to check on the character's feet 
		pixelsToCheckY = yPos + 79
		
		#make the pixel to check slightly in front of the player
		# facing left
		if self.direction == -1:
			pixelsToCheckx = xPos - 10
		# facing right
		else:
			pixelsToCheckx = xPos + 50

		# initial check the block 
		blockCheck = self.checkTerrain(pixelsToCheckx, pixelsToCheckY, world_data)

		# if the block not a gap to jump
		if blockCheck != -3:
			return blockCheck

		# if the block is a gap set teh pixel to check slight closer so that the player jumps later
		pixelsToCheckx = xPos + 25

		# Second pit check, closer to player
		Height = self.checkTerrain(pixelsToCheckx, pixelsToCheckY, world_data)
		return Height
		
	# check terrain type in front of player and return height of terrain relative to player
	def checkTerrain(self, pixelsToCheckx, pixelsToCheckY,world_data):
		height = 0
		tileCount = 20
	
		# convert pixels coordinates -> tile coords
		xCord = math.floor(pixelsToCheckx/50)
		yCord = math.floor(pixelsToCheckY/50)

		#boundary protection
		if xCord <0 or yCord<0 or xCord >=tileCount or yCord>=tileCount:
			return 5
		
		# the tile in front of us
		tileData = world_data[yCord][xCord] 
		# get Terrain Type

		# check the block in front of us
		# if the block is air, enemy spawn or coin position, move down a block to check
		if tileData == 0 or tileData == 3 or tileData == 7:  
			for i in range (1,4):
				#add 1 to height to check tile above
				yCord += 1 
				#set height to the -index of the loop
				height = (-i) 
				#get tile data
				tileData = world_data[yCord][xCord] 

				#get Terrain again
				if tileData == 1 or tileData == 2: 
					break
				#if the tile is lava set height to -3 
				elif tileData == 6:
					height = -3
					break

		#CHECKS ABOVE US
		# if the block is a dirt block, move up a block to check
		elif tileData == 1 or tileData == 2:
			for i in range (1,4):
				# subtract 1 from yCord to check tile above
				yCord -= 1
				# set height to the index of the loop
				height = i
				# get tile data
				tileData = world_data[yCord][xCord]
				
				#get Terrain again
				# if the tile above is air, enemy spawn or coin position, stop checking
				if tileData == 0 or tileData == 3 or tileData == 7:
					break

		return height
							
	# Reset player position
	def reset(self):
		self.images_right = []
		self.images_left = []
		self.index = 0
		self.counter = 0
		self.dx = 0
		self.dy = 0
		
		# draw animation for the player
		for num in range(1, 5):
			img_right = pygame.image.load(f'./ThePlatformerGame/img/guy{num}.png')
			img_right = pygame.transform.scale(img_right, (40*scale, 80*scale))
			img_left = pygame.transform.flip(img_right, True, False)
			self.images_right.append(img_right)
			self.images_left.append(img_left)
		
		self.dead_image = pygame.image.load('./ThePlatformerGame/img/ghost.png')

		# reset player default position
		self.image = self.images_right[self.index]
		self.rect = self.image.get_rect()
		self.rect.x = 100
		self.rect.y = screen_height - 130
		self.width = self.image.get_width()
		self.height = self.image.get_height()
		self.vel_y = 0
		self.jumped = False
		self.direction = 1
		self.in_air = True

# World class
class World():
	# Initialize world with level data
	def __init__(self, data):
		self.tile_list = []

		# create sprite groups
		self.blob_group = pygame.sprite.Group()
		self.lava_group = pygame.sprite.Group()
		self.coin_group = pygame.sprite.Group()
		self.exit_group = pygame.sprite.Group()

		#load images
		dirt_img = pygame.image.load('./ThePlatformerGame/img/dirt.png')
		grass_img = pygame.image.load('./ThePlatformerGame/img/grass.png')

		row_count = 0

		# for each value in the world data and create the corresponding tile/enemy/coin at that position 
		# 2d for loop
		for row in data:
			col_count = 0
			for tile in row:
				# if tile is dirt
				if tile == 1:
					img = pygame.transform.scale(dirt_img, (tile_size, tile_size))
					img_rect = img.get_rect()
					img_rect.x = col_count * tile_size
					img_rect.y = row_count * tile_size
					tile = (img, img_rect)
					self.tile_list.append(tile)
				
				# if tile is grass
				if tile == 2:
					img = pygame.transform.scale(grass_img, (tile_size, tile_size))
					img_rect = img.get_rect()
					img_rect.x = col_count * tile_size
					img_rect.y = row_count * tile_size
					tile = (img, img_rect)
					self.tile_list.append(tile)
				
				# if tile is enemy
				if tile == 3:
					blob = Enemy(col_count * tile_size, row_count * tile_size + 15)
					self.blob_group.add(blob)
				
				# if tile is lava
				if tile == 6:
					lava = Lava(col_count * tile_size, row_count * tile_size + (tile_size // 2))
					self.lava_group.add(lava)
				
				# if tile is coin
				if tile == 7:
					coin = Coin(col_count * tile_size + (tile_size // 2), row_count * tile_size + (tile_size // 2))
					self.coin_group.add(coin)
				
				# if tile is goal
				if tile == 8:
					exit = Exit(col_count * tile_size, row_count * tile_size - (tile_size // 2))
					self.exit_group.add(exit)
				col_count += 1
			row_count += 1

	# Draw the world
	def draw(self):
		for tile in self.tile_list:
			screen.blit(tile[0], tile[1])
			#shows hitboxes
			if show_hitboxes:
				pygame.draw.rect(screen, (255, 255, 255), tile[1], 2)

# Enemy class
class Enemy(pygame.sprite.Sprite):
	# Initialize enemy
	def __init__(self, x, y):
		pygame.sprite.Sprite.__init__(self)
		img = pygame.image.load('./ThePlatformerGame/img/blob.png')
		# scale enemy size
		self.image = pygame.transform.scale(img, (tile_size-1, tile_size-tile_size*0.3))
		self.rect = self.image.get_rect()
		
		# set enemy position
		self.rect.x = x
		self.rect.y = y-((tile_size/3)-tile_size*0.3)
		
		# randomly choose left or right direction at beginning
		if pygame.time.get_ticks() % 2 == 0:
			self.move_direction = 1
		else:
			self.move_direction = -1

		self.move_counter = 0

	# Update enemy position
	def update(self):
		if slowerEnemy:
			if self.move_counter %2 == 0:
				self.rect.x += self.move_direction
			self.move_counter += 1
		else:
			self.rect.x += self.move_direction
			self.move_counter += 1
		
		# change direction if has moved 50 pixels
		if abs(self.move_counter) > 50*scale:
			self.move_direction *= -1
			self.move_counter *= -1

# Lava class
class Lava(pygame.sprite.Sprite):
	# Initialize lava
	def __init__(self, x, y):
		pygame.sprite.Sprite.__init__(self)
		img = pygame.image.load('./ThePlatformerGame/img/lava.png')
		self.image = pygame.transform.scale(img, (tile_size, tile_size // 2))
		self.rect = self.image.get_rect()
		# set lava position
		self.rect.x = x
		self.rect.y = y

# Coin class
class Coin(pygame.sprite.Sprite):
	# Initialize coin
	def __init__(self, x, y):
		pygame.sprite.Sprite.__init__(self)
		img = pygame.image.load('./ThePlatformerGame/img/coin.png')
		self.image = pygame.transform.scale(img, (tile_size // 2, tile_size // 2))
		self.rect = self.image.get_rect()
		# set coin position
		self.rect.center = (x, y)

# Exit class
class Exit(pygame.sprite.Sprite):
	# Initialize exit
	def __init__(self, x, y):
		pygame.sprite.Sprite.__init__(self)
		img = pygame.image.load('./ThePlatformerGame/img/exit.png')
		self.image = pygame.transform.scale(img, (tile_size, int(tile_size * 1.5)))
		self.rect = self.image.get_rect()
		
		# set exit position
		self.rect.x = x
		self.rect.y = y

# Main function to run either human or pathfinding AI
def humanOrPathfind(use_ai):
	run = True
	platformE = platformerEnv()

	# Main game loop
	while run:
		global score
		# set framerate
		clock.tick(fps)

		# draw background
		screen.blit(bg_img, (0, 0))
		screen.blit(sun_img, (100, 100))
		platformE.world.draw()
		
		# if the game is not over
		if platformE.game_over == 0:
			# player has won reset timer
			if platformE.gameWon >= 1:
				current_time = platformE.wonTime

			# clacue time played and draw to screen
			current_time = pygame.time.get_ticks() - platformE.start_time
			seconds = current_time // 1000
			milliseconds = current_time % 1000
			timer_text = f"Time: {seconds}.{milliseconds:03d}"

			platformE.draw_text(' X ' + str(score), font_score, black, ((tile_size-5)*scale), (8*scale))
			platformE.draw_text(timer_text, font_score, black, (screen_width-200)*scale, 4*scale)

			# move enemies
			platformE.world.blob_group.update()

			#check if a coin has been collected and update score
			if pygame.sprite.spritecollide(platformE.player, platformE.world.coin_group, True):
				score += 1
				
			# draw sprites
			platformE.world.blob_group.draw(screen)
			platformE.world.lava_group.draw(screen)
			platformE.world.coin_group.draw(screen)
			platformE.world.exit_group.draw(screen)
			#test platformE.player.getTerrainInFront()

			# Use AI or player control based on selected mode
			if use_ai:
				# Get next AI action from ai_pathfinding.py
				from ai_pathfinding import terrain_ai
				action = terrain_ai(platformE)
			else:
				action = 10  # Use 10 to indicate player control
			
			#test print("AI Action:", action)
			platformE.game_over = platformE.player.update(action, platformE.world, platformE.game_over)

			#if player has died
			if platformE.game_over == -1:
					# reset world and player
					platformE.player.reset()
					platformE.game_over = 0
					score = 0
					platformE.world = platformE.reset_level()
					
			#if player reaches exit
			if platformE.game_over == 1:
				# reset world, player and time
				platformE.player.reset()
				platformE.game_over = 0
				score = 0
				print("Level Complete! Time taken:", seconds, "seconds and", milliseconds, "milliseconds")
				platformE.start_time = pygame.time.get_ticks()
				platformE.gameWon += 1
				platformE.world = platformE.reset_level()
				
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
		# debug print
		#platformE.d14Vector()
		
		pygame.display.update()

	pygame.quit()
	sys.exit()