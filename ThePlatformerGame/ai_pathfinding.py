import math



## todo: when in air only input the direct moving instead of the direction with jump





# Action mapping:
# 0 left, 1 left+jump, 2 right, 3 right+jump, 4 up, 5 idle

LOOKAHEAD = 1  # just check one tile ahead like your system


def terrain_ai(platform_env):
	global world_data
	player = platform_env.player
	world_data = platform_env.world_data
	tile_size = platform_env.tile_size

	# Get target (coin or exit)
	coins = list(platform_env.world.coin_group)
	exits = list(platform_env.world.exit_group)
	target = closest_sprite(player, coins) if coins else closest_sprite(player, exits)

	# Decide direction toward closest target (coin or exit)
	if target:
		if target.rect.centerx > player.rect.centerx:
			playerDir = 1   # target is to the right
		else:
			playerDir = -1  # target is to the left
	else:
		playerDir = player.direction  # fallback when no target found
	# 👇 Call our duplicated terrain check
	height = getTerrainInFront(player)

	slimeD, slimeX, slime_direction = getClosestEnemyDistance(platform_env, player)

	slime_moving_toward_player = False
	slime_moving_away_from_player = False

	
	if slimeD <= 65:

		# Slime moving right AND slime is left of player = moving toward
		if slime_direction == 1 and slimeX < player.rect.centerx:
			slime_moving_toward_player = True

		# Slime moving left AND slime is right of player = moving toward  
		if slime_direction == -1 and slimeX > player.rect.centerx:
			slime_moving_toward_player = True

		# Slime moving right AND slime is right of player = moving away
		if slime_direction == 1 and slimeX > player.rect.centerx:
			slime_moving_away_from_player = True

		# Slime moving left AND slime is left of player = moving away
		if slime_direction == -1 and slimeX < player.rect.centerx:
			slime_moving_away_from_player = True

		# React to slimes moving toward us
		if slime_moving_toward_player and slimeD <= 65:
			if slimeX < player.rect.centerx:  # slime is left
				return 3  # Right + Jump
			else:  # slime is right
				return 1  # Left + Jump

		# React to slimes moving away: maintain distance / move opposite
		if slime_moving_away_from_player and slimeD <= 65:
			if slimeX > player.rect.centerx:  # slime is left
				return 0  # move left away
			else:  # slime is right
				return 2  # move right away
			

	# For slimes not moving or farther than 100px, continue normal terrain logic

	if playerDir == 1:  # facing right
		#gap
		if height==-1:
			return 2
		if height==-2:
			return 2
		if height==-3:
			return 3
		#jump
		if height==1:
			return 3
		if height==2:
			return 3
		return 5  # Default fallback
	elif playerDir == -1: # facing left
		#gap
		if height==-1:
			return 0
		if height==-2:
			return 0
		if height==-3:
			return 1
		#jump
		if height==1:
			return 1
		if height==2:
			return 1
		return 5  # Default fallback
	return 5  # Default fallback


def getTerrainInFront(player):
	# Player position
	xPos = player.rect.x
	yPos = player.rect.y

	pixelsToCheckY = yPos + 79
	
	#facing left
	if player.direction == -1:
		pixelsToCheckx = xPos - 10
	#facing right
	else:
		pixelsToCheckx = xPos + 50

	blockCheck = checkTerrain(player, pixelsToCheckx, pixelsToCheckY, world_data)

	# if no gap detected
	if blockCheck != -3:
		# facing left
		if player.direction == -1:   
			pixelsToCheckx = xPos - 10
		# facing right
		elif player.direction == 1: 
			pixelsToCheckx = xPos + 50

		Height = checkTerrain(player, pixelsToCheckx, pixelsToCheckY, world_data)
		return Height

	
	pixelsToCheckx = xPos + 25

	# Second pit check, closer to player
	Height = checkTerrain(player, pixelsToCheckx, pixelsToCheckY, world_data)
	return Height

	#return relative height
	#3 big wall
	#2 2 block
	#1 1 block
	#-1 next block along
	#-2 1 block gap
	#-3 2 block gap
	

def checkTerrain(player, pixelsToCheckx, pixelsToCheckY,world_data):
	height = 0
	tileCount = 20
	
	#convert pixels coordinates -> tile coords
	xCord = math.floor(pixelsToCheckx/50)
	yCord = math.floor(pixelsToCheckY/50)


	#boundary protection
	if xCord <0 or yCord<0 or xCord >=tileCount or yCord>=tileCount:
		return 5
	

	tileData = world_data[yCord][xCord] #the tile in front of us
	Terrain = getTerrain(tileData) #get Terrain Type


	#CHECKS BELOW US

	if Terrain == 0 or Terrain == 2 or Terrain == 3: #if a air, enemy or objective   
		for i in range (1,4):
			yCord += 1 #add 1 to height to check tile above
			height = (-i) #set height to the -index of the loop
			tileData = world_data[yCord][xCord] #get tile data

			if getTerrain(world_data[yCord][xCord]) == 1: #get Terrain again
				break
			elif getTerrain(world_data[yCord][xCord]) == 4:
				height = -3
				break

	#CHECKS ABOVE US

	elif Terrain == 1:
		for i in range (1,4):
			yCord -= 1
			height = i
			tileData = world_data[yCord][xCord]
			TerrainAbove = getTerrain(tileData)
			if TerrainAbove == 0 or TerrainAbove == 2 or TerrainAbove == 3:
				break

	return height
						
def getTerrain(tileData):					
	if tileData == 0:
		Terrain = 0 #there is a gap needs jumped
	elif tileData == 1 or tileData == 2:
		Terrain = 1 #there is a block, walk forwards
	elif tileData == 3:
		Terrain = 2 #there is slime
	elif tileData == 6:
		Terrain = 4 # there is lava
	else:
		Terrain = 3 #there is objective

	return Terrain

def closest_sprite(player, sprites):
	min_dist = float('inf')
	closest = None
	for s in sprites:
		dist = math.hypot(
			player.rect.centerx - s.rect.x,
			(player.rect.centery - s.rect.y)*2,
		)
		if dist < min_dist:
			min_dist = dist
			closest = s
	return closest


def getClosestEnemyDistance(platform_env, player):
	playerX = player.rect.centerx
	playerY = player.rect.y+55
	minDistance = float('inf')
	closestEnemyX = playerX
	closestSlimeDirection = None
	
	for enemy in platform_env.world.blob_group:
		enemyX = enemy.rect.centerx
		enemyY = enemy.rect.y+12
		#Slime height = 52, player height = 80
		distance = ((enemyX - playerX) ** 2 + ((enemyY) - (playerY)) ** 2) ** 0.5
		if distance < minDistance:
			minDistance = distance
			closestEnemyX = enemyX
			# Get direction of the CLOSEST slime
			if hasattr(enemy, 'move_direction'):
				closestSlimeDirection = enemy.move_direction
			else:
				closestSlimeDirection = 1  # Default direction
	
	return minDistance, closestEnemyX, closestSlimeDirection
