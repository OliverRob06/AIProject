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
	def __init__(self):
		
		#Initialize Environment with default attributes
		self.player = Player()

		#create dummy coin for showing the score
		self.start_ticks = pygame.time.get_ticks()
		self.score_coin = Coin(tile_size // 2, tile_size // 2)
		self.world = World(world_data)
		self.world.coin_group.add(self.score_coin)

		#create buttons
		self.restart_button = Button(screen_width // 2 - 50, screen_height // 2 + 100, restart_img)
		self.start_button = Button(screen_width // 2 - 350, screen_height // 2, start_img)
		self.exit_button = Button(screen_width // 2 + 150, screen_height // 2, exit_img)
		self.win_button = Button(screen_width //2 - 50, screen_height// 2, win_img)

		self.start_time = pygame.time.get_ticks()
		self.gameWon = False
		self.wonTime = 0	
		self.frame = 0
		self.game_over = 0
		

	def draw_text(self,text, font, text_col, x, y):
		img = font.render(text, True, text_col)
		screen.blit(img, (x, y))
		
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

	def d14Vector(self):
		playerX = self.player.rect.x
		playerY = self.player.rect.y

		key = pygame.key.get_pressed()
		left = key[pygame.K_LEFT]
		right = key[pygame.K_RIGHT]
		space = key[pygame.K_SPACE]
		
		if not left and not right and not space:
			nothing = True
		else:
			nothing = False

		playerInAir = self.player.in_air 

		playerVelY = self.player.rect.y - (self.player.rect.y - self.player.vel_y)

		# see terrain infront of player
		# later

		# enemies coords
		# later

		closestEnemyDistance = self.getClosestEnemyDistance(self.player.rect.x, self.player.rect.y)
		closestGoalOrCoin = self.getClosestGoalOrCoinDistance(self.player.rect.x, self.player.rect.y)

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
		for enemy in self.world.blob_group:
			enemyX = enemy.rect.x
			enemyY = enemy.rect.y
		print("Enemy X: ", enemyX)
		print("Enemy Y: ", enemyY)

		print("Closest Enemy Distance: ", closestEnemyDistance)
		print("Closest Goal/Coin Distance: ", closestGoalOrCoin)
		print("Time Spent (ms): ", timeSpent)

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
	
	def getClosestGoalOrCoinDistance(self, playerX, playerY):
		minDistance = float('inf')
		for coin in self.world.coin_group:
			coinX = coin.rect.x
			coinY = coin.rect.y
			distance = ((coinX - playerX +45 - 80) ** 2 + ((coinY+52) - (playerY+80)) ** 2) ** 0.5
			if distance < minDistance:
				minDistance = distance
		for goal in self.world.exit_group:
			goalX = goal.rect.x
			goalY = goal.rect.y
			distance = ((goalX - playerX +45 - 80) ** 2 + ((goalY+52) - (playerY+80)) ** 2) ** 0.5
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

	# Steal d14Vecotr code when done
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

	# Inside platformerEnv
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

		# Adds player distance to nearest coin
		state_vector.append()


		# Add terrain infront of players relative height to player
		# state_vector.append(self.player.getNearestSurface())

		# Add distance to nearest enemy
		state_vector.append(self.getClosestEnemyDistance(self.player.rect.x, self.player.rect.y))

		# Add distance to nearest coin or goal
		state_vector.append(self.getClosestGoalOrCoinDistance(self.player.rect.x, self.player.rect.y))


		# Returns Set turned into NumPy Float Tensor 
		return np.array(state_vector, dtype=np.float32)

		

class Button():
	def __init__(self, x, y, image):
		self.image = image
		self.rect = self.image.get_rect()
		self.rect.x = x
		self.rect.y = y
		self.clicked = False

	def draw(self):
		action = False

		#get mouse position
		pos = pygame.mouse.get_pos()

		#check mouseover and clicked conditions
		if self.rect.collidepoint(pos):
			if pygame.mouse.get_pressed()[0] == 1 and self.clicked == False:
				action = True
				self.clicked = True

		if pygame.mouse.get_pressed()[0] == 0:
			self.clicked = False


		#draw button
		screen.blit(self.image, self.rect)

		return action

class Player():
	def __init__(self):
		self.reset()


	# action corresponds to either 10 for player input, or 0-5 for ai input
	def update(self, action, world, game_over):
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
			if (action<7) :
				if (action==4 or action==3 or action==4) and self.jumped == False and self.in_air == False:
					self.vel_y = -15
					self.jumped = True
				if (action==4 or action==3 or action==4) == False:
					self.jumped = False
				if (action==0 or action==1):
					self.dx -= 5*scale
					self.counter += 1
					self.direction = -1
				if (action==2 or action==3):
					self.dx += 5*scale
					self.counter += 1
					self.direction = 1
				if (action==0 or action==1) == False and (action==2 or action==3) == False:
					self.counter = 0
					self.index = 0
					if self.direction == 1:
						self.image = self.images_right[self.index]
					if self.direction == -1:
						self.image = self.images_left[self.index]
			else:
				
				key = pygame.key.get_pressed()
				if key[pygame.K_SPACE] and self.jumped == False and self.in_air == False:
					self.vel_y = -15
					self.jumped = True
				if key[pygame.K_SPACE] == False:
					self.jumped = False
				if key[pygame.K_LEFT]:
					self.dx -= 5*scale
					self.counter += 1
					self.direction = -1
				if key[pygame.K_RIGHT]:
					self.dx += 5*scale
					self.counter += 1
					self.direction = 1
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

	def reset(self):
		self.images_right = []
		self.images_left = []
		self.index = 0
		self.counter = 0
		for num in range(1, 5):
			img_right = pygame.image.load(f'img/guy{num}.png')
			img_right = pygame.transform.scale(img_right, (40*scale, 80*scale))
			img_left = pygame.transform.flip(img_right, True, False)
			self.images_right.append(img_right)
			self.images_left.append(img_left)
		self.dead_image = pygame.image.load('img/ghost.png')
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
	def __init__(self, data):
		self.tile_list = []

		self.blob_group = pygame.sprite.Group()
		self.lava_group = pygame.sprite.Group()
		self.coin_group = pygame.sprite.Group()
		self.exit_group = pygame.sprite.Group()

		#load images
		dirt_img = pygame.image.load('img/dirt.png')
		grass_img = pygame.image.load('img/grass.png')

		row_count = 0
		for row in data:
			col_count = 0
			for tile in row:
				# dirt
				if tile == 1:
					img = pygame.transform.scale(dirt_img, (tile_size, tile_size))
					img_rect = img.get_rect()
					img_rect.x = col_count * tile_size
					img_rect.y = row_count * tile_size
					tile = (img, img_rect)
					self.tile_list.append(tile)
				# grass
				if tile == 2:
					img = pygame.transform.scale(grass_img, (tile_size, tile_size))
					img_rect = img.get_rect()
					img_rect.x = col_count * tile_size
					img_rect.y = row_count * tile_size
					tile = (img, img_rect)
					self.tile_list.append(tile)
				# enemy
				if tile == 3:
					blob = Enemy(col_count * tile_size, row_count * tile_size + 15)
					self.blob_group.add(blob)
				# lava
				if tile == 6:
					lava = Lava(col_count * tile_size, row_count * tile_size + (tile_size // 2))
					self.lava_group.add(lava)
				# coin
				if tile == 7:
					coin = Coin(col_count * tile_size + (tile_size // 2), row_count * tile_size + (tile_size // 2))
					self.coin_group.add(coin)
				# goal
				if tile == 8:
					exit = Exit(col_count * tile_size, row_count * tile_size - (tile_size // 2))
					self.exit_group.add(exit)
				col_count += 1
			row_count += 1

	def draw(self):
		for tile in self.tile_list:
			screen.blit(tile[0], tile[1])
			#shows hitboxes
			if show_hitboxes:
				pygame.draw.rect(screen, (255, 255, 255), tile[1], 2)

class Enemy(pygame.sprite.Sprite):
	def __init__(self, x, y):
		pygame.sprite.Sprite.__init__(self)
		img = pygame.image.load('img/blob.png')
		self.image = pygame.transform.scale(img, (tile_size-1, tile_size))
		self.rect = self.image.get_rect()
		self.rect.x = x

		self.rect.y = y-(tile_size/3)
		
		self.move_direction = 1
		self.move_counter = 0

	def update(self):
		if slowerEnemy:
			if self.move_counter %2 == 0:
				self.rect.x += self.move_direction
			self.move_counter += 1
		else:
			self.rect.x += self.move_direction
			self.move_counter += 1
		
		
		if abs(self.move_counter) > 50*scale:
			self.move_direction *= -1
			self.move_counter *= -1

class Lava(pygame.sprite.Sprite):
	def __init__(self, x, y):
		pygame.sprite.Sprite.__init__(self)
		img = pygame.image.load('img/lava.png')
		self.image = pygame.transform.scale(img, (tile_size, tile_size // 2))
		self.rect = self.image.get_rect()
		self.rect.x = x
		self.rect.y = y

class Coin(pygame.sprite.Sprite):
	def __init__(self, x, y):
		pygame.sprite.Sprite.__init__(self)
		img = pygame.image.load('img/coin.png')
		self.image = pygame.transform.scale(img, (tile_size // 2, tile_size // 2))
		self.rect = self.image.get_rect()
		self.rect.center = (x, y)

class Exit(pygame.sprite.Sprite):
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
			start_time = pygame.time.get_ticks()
			platformE.gameWon += 1
			platformE.world = platformE.reset_level()
			
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False

	if platformE.frame % 10 == 0:
		platformE.d14Vector()

	platformE.frame += 1
	pygame.display.update()

pygame.quit()