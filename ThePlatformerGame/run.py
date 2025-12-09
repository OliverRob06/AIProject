import platformer_tut8 as pathPlayer
import train_agent as trainA
# Choice menu
print("=" * 50)
print("PLATFORMER GAME - MODE SELECTION")
print("=" * 50)
print("1. Player Control (Use Arrow Keys & Space)")
print("2. AI Pathfinding (Watch AI complete level)")
print("3. Train Machine Learning AI")
print("=" * 50)

mode = None
while mode not in [1, 2, 3]:
	try:
		mode = int(input("Select mode (1, 2, or 3): "))
		if mode not in [1, 2]:
			print("Invalid choice. Please enter 1, 2 or 3.")
	except ValueError:
		print("Please enter a valid number (1, 2 or 3).")

if mode == 1:
	use_ai = False
	print("\nStarting in PLAYER CONTROL mode...")
	pathPlayer.humanOrPathfind(use_ai)
if mode == 2:
	use_ai = True
	print("\nStarting in AI PATHFINDING mode...")
	print("Watch as the AI finds and follows the optimal path to the goal!")
	pathPlayer.humanOrPathfind(use_ai)
else:
	print("\nStarting in AI Training mode...")
	trainA.agentStart()


