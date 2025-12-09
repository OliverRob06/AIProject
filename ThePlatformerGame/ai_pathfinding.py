import math

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

    dx = target.rect.centerx - player.rect.centerx
    dy = target.rect.centery - player.rect.centery

    # If coin is above player, jump towards it
    if dy < -50:  # Coin is significantly above
        if dx > 0:
            return 3  # Jump right
        else:
            return 1  # Jump left

    # Set direction before checking terrain
    player.direction = 1 if dx > 0 else -1

    # 👇 Call our duplicated terrain check
    height = getTerrainInFront(player)

    # same logic as your terrain function uses
    if player.direction == 1:  # facing right
        if height==-1:
            return 2
        if height==-2:
            return 2
        if height==-3:
            return 3

        if height==1:
            return 3
        if height==2:
            return 3
        return 5  # Default fallback
    elif player.direction == -1: # facing left
        if height==-1:
            return 0
        if height==-2:
            return 0
        if height==-3:
            return 1

        if height==1:
            return 1
        if height==2:
            return 5
        return 5  # Default fallback
    
    return 5


def getTerrainInFront(player):
		#tile size = 50px^2
		#dirt block = 1, grass block = 2, enemy = 3, lava = 6, coin = 7, goal = 8
		#player coords
		xPos = player.rect.x #100 by default
		yPos = player.rect.y #870 by default


		#find block below

		#direction player faces
		if player.direction == -1:  #facing left
			pixelsToCheckx = xPos+25 #block on left
			pixelsToCheckY = yPos+79
			Height = checkTerrain(player, pixelsToCheckx, pixelsToCheckY, world_data)
		else:
			pixelsToCheckx = xPos+50#block on right
			pixelsToCheckY = yPos+79
			Height = checkTerrain(player, pixelsToCheckx, pixelsToCheckY,world_data)
		
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
            player.rect.centerx - s.rect.centerx,
            (player.rect.centery - s.rect.centery)*1.5,
        )
        if dist < min_dist:
            min_dist = dist
            closest = s
    return closest

