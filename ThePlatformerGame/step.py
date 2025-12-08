# AI takes action and returns new state and reward associated
def step(self, action):
    # Handle Pygame events (keep window from freezing)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    # Translate the given number corresponding to an action, and subsequent movement.
    self.game_over = self.player.update(self.game_over, action, self.world)
    
    # For calculating reward
    # Starts at -0.2 to incentivise taking less time
    reward = -0.2
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