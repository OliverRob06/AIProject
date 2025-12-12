import math

# verbose data
# intial pixel to check
# height reutrn
# pixel to check after encounterning a gap (-3)
# action
# 
# coin distance changing
# low distance == happy
# high distance == the ignore method
# 
# slime when within certain distance
# moving away or towards 

# Action mapping:
# 0 left, 1 left+jump, 2 right, 3 right+jump, 4 up, 5 idle

def terrain_ai(platform_env):
	global world_data
	player = platform_env.player
	world_data = platform_env.world_data

	# Get target (coin or exit)
	coins = list(platform_env.world.coin_group)
	exits = list(platform_env.world.exit_group)

	target = closest_sprite(player, coins) if coins else closest_sprite(player, exits)

	# Decide direction toward closest target (coin or exit)
	if target:
		if target.rect.centerx > player.rect.centerx:
			# target is to the right
			playerDir = 1   
		else:
			# target is to the left
			playerDir = -1  
	else:
		# fallback when no target found
		playerDir = player.direction  
	# 👇 Call our duplicated terrain check
	height = moveCheckPixelX(player)

	# get slime distance x and direction
	slimeD, slimeX, slime_direction = getClosestEnemyDistance(platform_env, player)

	# initialize slime movement relative to player
	slime_moving_toward_player = False
	slime_moving_away_from_player = False
	
	#if slime is with 65px 
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
				if playerDir == 1:  # facing right
					return 2
				else:
					return 3  # Right + Jump
			else:  # slime is right
				if playerDir == -1:  # facing left
					return 0
				else:
					return 1  # Left + Jump

		# React to slimes moving away: maintain distance / move opposite
		if slime_moving_away_from_player and slimeD <= 65:
			if slimeX > player.rect.centerx:  # slime is left
				return 0  # move left away
			else:  # slime is right
				return 2  # move right away
			
	# For slimes not moving or farther than 100px, continue normal terrain logic
	if playerDir == 1:  # facing right
		# if there is a a block the player can walk/fall onto move normally
		if height==-1 or height==-2:
			return 2
		
		# if the block in front is a pit or lava jump
		if height==-3:
			# if the player is already in air continue moving forward
			if player.in_air:
				return 2
			
			# else jump
			else:
				return 3
		#if the block is required to be jumped onto
		if height==1 or height==2:
			# if the player is already in air continue moving forward
			if player.in_air:
				return 2
			
			# else jump
			else:
				return 3
		return 5  # Default fallback
	
	elif playerDir == -1: # facing left
		# if there is a a block the player can walk/fall onto move normally
		if height==-1 or height==-2:
			return 0
		
		# if the block in front is a pit or lava jump
		if height==-3:
			# if the player is already in air continue moving forward
			if player.in_air:
				return 0
			
			# else jump
			else:
				return 1
			
		# if the block is required to be jumped onto
		if height==1 or height==2:
			# if the player is already in air continue moving forward
			if player.in_air:
				return 0
			
			# else jump
			else:
				return 1
		return 5  # Default fallback
	return 5  # Default fallback

def moveCheckPixelX(player):
	# Player position
	xPos = player.rect.x
	yPos = player.rect.y

	# Pixels to check on the character's feet 
	pixelsToCheckY = yPos + 79
	
	#make the pixel to check slightly in front of the player
	# facing left
	if player.direction == -1:
		pixelsToCheckx = xPos - 10
	# facing right
	else:
		pixelsToCheckx = xPos + 50

	# initial check the block 
	blockCheck = checkTerrain(player, pixelsToCheckx, pixelsToCheckY, world_data)

	originalCheckX = pixelsToCheckx

	# if the block not a gap to jump
	if blockCheck != -3:
		return blockCheck

	# if the block is a gap set teh pixel to check slight closer so that the player jumps later
	pixelsToCheckx = xPos + 25

	# Second pit check, closer to player
	Height = checkTerrain(pixelsToCheckx, pixelsToCheckY, world_data)
	return Height

def checkTerrain(pixelsToCheckx, pixelsToCheckY,world_data):
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
						
# get the disance to the closest sprite in a group
def closest_sprite(player, sprites):
	# initialize minimum distance to a large value
	min_dist = float('inf')
	# initialize closest sprite to None
	closest = None

	#for each sprite in the group
	for s in sprites:
		# calculate the distance to the player, favouring the x-axis
		dist = math.hypot(
			player.rect.centerx - s.rect.x,
			(player.rect.centery - s.rect.y)*2.5,
		)
		# if this distance is less than the minimum distance found so far
		if dist < min_dist:
			# update minimum distance and closest sprite
			min_dist = dist
			closest = s

	# return the closest sprite found
	return closest

# get the distance to the closest enemy (slime), the x position and its direction
def getClosestEnemyDistance(platform_env, player):
	# Player position
	playerX = player.rect.centerx
	playerY = player.rect.y+55

	# Initialize minimum distance and closest enemy position
	minDistance = float('inf')
	# Initialize closest enemy position
	closestEnemyX = playerX
	# Initialize closest slime direction
	closestSlimeDirection = None
	
	# For each enemy in the blob group
	for enemy in platform_env.world.blob_group:
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
			closestEnemyX = enemyX

			# Get direction of the CLOSEST slime if it has a move_direction attribute
			if hasattr(enemy, 'move_direction'):
				closestSlimeDirection = enemy.move_direction
			else:
				closestSlimeDirection = 1  # Default direction

	# return the minimum distance, closest enemy x position, and closest slime direction
	return minDistance, closestEnemyX, closestSlimeDirection
