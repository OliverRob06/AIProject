import pygame # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
from pygame.locals import * # pyright: ignore[reportMissingImports]

pygame.init()		
        
#						The Environment Global Variables
scale = 0.7
slowerEnemy = False
clock = pygame.time.Clock()
fps = 60

screen_width = 1000*scale
screen_height = 1000*scale

show_hitboxes = False

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Platformer')

#define game variables
tile_size = 50*scale
main_menu = True
score = 0

#define font variables
font_score = pygame.font.SysFont('Bauhaus 93', 30)

#define colours
black = (0, 0,0)

#load images
sun_img = pygame.image.load('img/sun.png')
bg_img = pygame.image.load('img/sky.png')
restart_img = pygame.image.load('img/restart_btn.png')
start_img = pygame.image.load('img/start_btn.png')
exit_img = pygame.image.load('img/exit_btn.png')
win_img = pygame.image.load('img/youwin.png')

#dirt block = 1, grass block = 2, enemy = 3, lava = 6, coin = 7, goal = 8
world_data = [
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0, 1], 
[1, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 8, 0, 0, 0, 2, 2, 0, 2, 2, 0, 0, 2, 2, 0, 2, 2, 0, 7, 1], 
[1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 1, 1], 
[1, 0, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 2, 2, 1, 1], 
[1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 2, 6, 6, 2, 2, 1, 1, 1, 1], 
[1, 0, 0, 2, 2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 
[1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 1, 0, 7, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 1], 
[1, 1, 2, 2, 0, 0, 3, 0, 0, 2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 1], 
[1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 7, 0, 1], 
[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 2, 1], 
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 2, 2, 2, 1, 1], 
[1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1], 
[1, 0, 0, 0, 0, 7, 0, 0, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 0, 0, 0, 2, 2, 6, 6, 1, 1, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]


class platformerEnv:
	# Initialize the environment
	def __init__(self):
		
		#Initialize Environment with default attributes
		self.player = Player()

		#create dummy coin for showing the score
		self.score_coin = Coin(tile_size // 2, tile_size // 2)
		self.world = World(world_data)
		self.world.coin_group.add(self.score_coin)

		self.start_time = pygame.time.get_ticks()
		self.gameWon = False
		self.wonTime = 0	
		self.frame = 0
		self.game_over = 0
	
	# Function to draw text on screen
	def draw_text(self, text, font, text_col, x, y):
		img = font.render(text, True, text_col)
		screen.blit(img, (x, y))

	# Reset the level	
	def reset_level(self):
		self.player.reset()
		self.world.blob_group.empty()
		self.world.coin_group.empty()
		self.world.lava_group.empty()
		self.world.exit_group.empty()

		world = World(world_data)
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
		# later

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
		closestGoalOrCoin = self.getClosestGoalOrCoinDistance(self.player.rect.x, self.player.rect.y)

		# time spent
		timeSpent = pygame.time.get_ticks() - self.start_time

		print("Player X: ", playerX)
		print("Player Y: ", playerY)

		print("key LEFT: ", left)
		print("key RIGHT: ", right)
		print("key SPACE: ", space)
		print("No Input: ", nothing)

		print("Player In Air: ", playerInAir)
		print("Player Vel Y: ", playerVelY)
		#Terrain

		print("Enemy X: ", enemyX)
		print("Enemy Y: ", enemyY)

		print("Closest Enemy Distance: ", closestEnemyDistance)
		print("Closest Goal/Coin Distance: ", closestGoalOrCoin)

		print("Goal x: ", exitX)
		print("Goal y: ", exitY)
		
		print("Player x: ", playX)
		print("Player y: ", playY)

		print("Time Spent (ms): ", timeSpent)

	# Get distance to closest enemy
	def getClosestEnemyDistance(self, playerX, playerY):
		minDistance = float('inf')
		for enemy in self.world.blob_group:
			enemyX = enemy.rect.x
			enemyY = enemy.rect.y
			#Slime height = 52, player height = 80
			distance = ((enemyX - playerX +45) ** 2 + ((enemyY+52) - (playerY+80)) ** 2) ** 0.5
			if distance < minDistance:
				minDistance = distance

		return minDistance
	
	# Get distance to closest goal or coin
	def getClosestGoalOrCoinDistance(self, playerX, playerY):
		minDistance = float('inf')
		for coin in self.world.coin_group:
			coinX = coin.rect.x
			coinY = coin.rect.y
			distance = ((coinX - playerX - 40) ** 2 + ((coinY+52) - (playerY+80)) ** 2) ** 0.5
			if distance < minDistance:
				minDistance = distance
		
		for goal in self.world.exit_group:
			goalX = goal.rect.x
			goalY = goal.rect.y
			distance = ((goalX + 35 - playerX) ** 2 + ((goalY+ - playerY)) ** 2) ** 0.5
			if distance < minDistance:
				minDistance = distance

		return minDistance

	## Reset, Step and getState all are in a class with player, world, enemies as elements
	# Used for the AI resetting
	def reset(self):
		self.player.reset()
		self.game_over = 0
		score = 0
		self.world = self.reset_level()
		
		# Return the initial state of the game and an empty info dict
		return self.get_state(), {}

	# Steal d14Vector code when done
	def step(self, action):
		# Handle Pygame events (keep window from freezing)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()

		# Translate the given number corresponding to an action, and subsequent movement.
		self.game_over = self.player.update(self.game_over, action, self.world)
		
		# Calculate reward
		reward = 0

		# Incentivises taking less time
		reward -= -0.2
		
		# If player has died
		if self.game_over == -1:
				reward-=100
				
		# If player reaches wins
		if self.game_over == 1:
			reward+=100
				
		# If player reaches coin (checkpoint)
		if pygame.sprite.spritecollide(self.player, self.coin_group, True):
			reward =+ 5 
			
		
		# Check game is over (Win or Lose)
		terminated = False
		if (self.game_over == -1 or self.game_over == 1):
			terminated = True

		# Draws Game. can remove to improve performance
		self.screen.fill((0,0,0))
		self.player.draw(self.screen)
		pygame.display.flip()

		# Return new observation given new state, reward calculated and game over
		return self.get_state(), reward, terminated, {}

	# Returns observation of current state
	def get_state(self):
		# create empty set for observation
		state_vector = []

		# Adding values to set that will become tensor/observation

		# Adds player x position 
		state_vector.append(self.player.rect.x)
		# Adds player y position 
		state_vector.append(self.player.rect.y)

		# Adds player in air boolean as 1 or 0 
		state_vector.append(int(self.player.in_air))
		# Adds player vertical velocity 
		state_vector.append(self.player.rect.y - (self.player.rect.y - self.player.vel_y))

		
		# Add terrain infront of players relative height to player
		# state_vector.append(self.player.getNearestSurface())

		# Add distance to nearest enemy
		state_vector.append(self.getClosestEnemyDistance(self.player.rect.x, self.player.rect.y))

		# Add distance to nearest coin or goal
		state_vector.append(self.getClosestGoalOrCoinDistance(self.player.rect.x, self.player.rect.y))


		# Returns Set turned into NumPy Float Tensor 
		return np.array(state_vector, dtype=np.float32)

class Player():
	# Initialize player
	def __init__(self):
		self.reset()

	# Fucntion to update player position (moving the player)
	def update(self, action, world, game_over):
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
		4. go up
		5: do nothing
		"""
		if game_over == 0:
			# Use AI actions for inputs
			if (action<10) :
				# AI jumping with varients (up left, up right and staight up)
				if (action==1 or action==3 or action==4) and self.jumped == False and self.in_air == False:
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

	# Reset player position
	def reset(self):
		self.images_right = []
		self.images_left = []
		self.index = 0
		self.counter = 0
		
		# draw animation for the player
		for num in range(1, 5):
			img_right = pygame.image.load(f'img/guy{num}.png')
			img_right = pygame.transform.scale(img_right, (40*scale, 80*scale))
			img_left = pygame.transform.flip(img_right, True, False)
			self.images_right.append(img_right)
			self.images_left.append(img_left)
		
		self.dead_image = pygame.image.load('img/ghost.png')

		# reset player default position
		self.image = self.images_right[self.index]
		self.rect = self.image.get_rect()
		self.rect.x = 100
		self.rect.y = screen_height - 130
		self.width = self.image.get_width()
		self.height = self.image.get_height()
		self.vel_y = 0
		self.jumped = False
		self.direction = 0
		self.in_air = True

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
		dirt_img = pygame.image.load('img/dirt.png')
		grass_img = pygame.image.load('img/grass.png')

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
				
				# coin
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

class Enemy(pygame.sprite.Sprite):
	# Initialize enemy
	def __init__(self, x, y):
		pygame.sprite.Sprite.__init__(self)
		img = pygame.image.load('img/blob.png')
		self.image = pygame.transform.scale(img, (tile_size-1, tile_size-tile_size*0.3))
		self.rect = self.image.get_rect()
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

class Lava(pygame.sprite.Sprite):
	# Initialize lava
	def __init__(self, x, y):
		pygame.sprite.Sprite.__init__(self)
		img = pygame.image.load('img/lava.png')
		self.image = pygame.transform.scale(img, (tile_size, tile_size // 2))
		self.rect = self.image.get_rect()
		self.rect.x = x
		self.rect.y = y

class Coin(pygame.sprite.Sprite):
	# Initialize coin
	def __init__(self, x, y):
		pygame.sprite.Sprite.__init__(self)
		img = pygame.image.load('img/coin.png')
		self.image = pygame.transform.scale(img, (tile_size // 2, tile_size // 2))
		self.rect = self.image.get_rect()
		self.rect.center = (x, y)

class Exit(pygame.sprite.Sprite):
	# Initialize exit
	def __init__(self, x, y):
		pygame.sprite.Sprite.__init__(self)
		img = pygame.image.load('img/exit.png')
		self.image = pygame.transform.scale(img, (tile_size, int(tile_size * 1.5)))
		self.rect = self.image.get_rect()
		self.rect.x = x
		self.rect.y = y


run = True
platformE = platformerEnv()
while run:

	

	clock.tick(fps)

	screen.blit(bg_img, (0, 0))
	screen.blit(sun_img, (100, 100))
	platformE.world.draw()
	
	if platformE.game_over == 0:
		if platformE.gameWon >= 1:
			current_time = platformE.wonTime

		current_time = pygame.time.get_ticks() - platformE.start_time
		seconds = current_time // 1000
		milliseconds = current_time % 1000
		timer_text = f"Time: {seconds}.{milliseconds:03d}"

		platformE.draw_text(' X ' + str(score), font_score, black, ((tile_size-5)*scale), (8*scale))
		platformE.draw_text(timer_text, font_score, black, (screen_width-200)*scale, 4*scale)

		platformE.world.blob_group.update()
		#update score
		#check if a coin has been collected
		if pygame.sprite.spritecollide(platformE.player, platformE.world.coin_group, True):
			score += 1
			
		
		platformE.world.blob_group.draw(screen)
		platformE.world.lava_group.draw(screen)
		platformE.world.coin_group.draw(screen)
		platformE.world.exit_group.draw(screen)
		platformE.game_over = platformE.player.update(10,platformE.world,platformE.game_over)

		#if player has died
		if platformE.game_over == -1:
				platformE.player.reset()
				platformE.game_over = 0
				score = 0
				platformE.world = platformE.reset_level()
				
		#if player reaches exit
		if platformE.game_over == 1:
			platformE.player.reset()
			platformE.game_over = 0
			score = 0
			platformE.start_time = pygame.time.get_ticks()
			platformE.gameWon += 1
			platformE.world = platformE.reset_level()
			
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False

	if platformE.frame % 20 == 0:
		platformE.d14Vector()

	platformE.frame += 1
	pygame.display.update()

pygame.quit()