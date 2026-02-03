env = ExecEnv("/Users/jackfletcher/Desktop/FYP_Data/2026-01-01_steps_5s.parquet")
obs, _ = env.reset()

done = False
total = 0
while not done:
    action = env.action_space.sample()
    obs, r, done, _, info = env.step(action)
    total += r

print("Return:", total)
print("Info:", info)
