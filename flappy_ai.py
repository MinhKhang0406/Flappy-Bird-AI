import gymnasium as gym
import flappy_bird_gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import argparse
import os

# ==========================================
# 1. XÂY DỰNG MẠNG NƠ-RON (BỘ NÃO CỦA AI)
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Mạng cơ bản với 2 lớp ẩn (Hidden layers)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. XÂY DỰNG AI AGENT (THUẬT TOÁN DEEP Q-LEARNING)
# ==========================================
class FlappyAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=50000) # Bộ nhớ kinh nghiệm
        
        # Các tham số học tập
        self.gamma = 0.99       # Hệ số giảm giá (quan tâm tương lai)
        self.epsilon = 1.0      # Tỷ lệ khám phá ban đầu (chơi ngẫu nhiên)
        self.epsilon_min = 0.01 # Tỷ lệ khám phá tối thiểu
        self.epsilon_decay = 0.995 # Tốc độ giảm sự ngẫu nhiên
        self.batch_size = 64
        self.lr = 0.001

        # Khởi tạo mạng chính và mạng mục tiêu
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Lưu lại kinh nghiệm sau mỗi bước nhảy"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_playing=False):
        """Hành động: Ngẫu nhiên (khám phá) hoặc Dựa trên kinh nghiệm (khai thác)"""
        if not is_playing and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        """Học lại từ những kinh nghiệm đã lưu trong bộ nhớ"""
        if len(self.memory) < self.batch_size:
            return

        # Lấy ngẫu nhiên một mẻ (batch) kinh nghiệm để học
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Q-value hiện tại
        current_q = self.model(states).gather(1, actions)
        
        # Q-value mục tiêu (Target)
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Cập nhật trọng số mạng
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Giảm dần sự ngẫu nhiên
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        print(f"💾 Đã lưu mô hình AI vào file: {filename}")

    def load(self, filename):
        if os.path.exists(filename):
            self.model.load_state_dict(torch.load(filename))
            self.model.eval() # Chuyển sang chế độ test
            self.epsilon = 0.0 # Bỏ qua ngẫu nhiên khi đã load file
            print(f"✅ Đã tải thành công file mô hình: {filename}")
        else:
            print(f"❌ Không tìm thấy file {filename}!")

# ==========================================
# 3. HÀM HUẤN LUYỆN (TRAIN)
# ==========================================
def train():
    env = gym.make("FlappyBird-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = FlappyAgent(state_dim, action_dim)
    
    episodes = 500 # Số vòng đời huấn luyện
    model_file = "flappy_brain.pth"

    print("🚀 BẮT ĐẦU HUẤN LUYỆN AI...")
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Khuyến khích sống sót, phạt nặng nếu chết
            if done: reward = -10
            elif action == 1: reward = 0.1 # Phạt nhẹ nếu nhảy vô tội vạ
            else: reward = 1 # Thưởng khi sống sót
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.replay()
            
            if done:
                agent.update_target_model()
                print(f"Vòng: {e+1}/{episodes} | Điểm sống: {total_reward:.1f} | Epsilon (Ngẫu nhiên): {agent.epsilon:.3f}")
                break
                
        # Lưu model mỗi 50 vòng
        if (e + 1) % 50 == 0:
            agent.save(model_file)

    agent.save(model_file)
    env.close()

# ==========================================
# 4. HÀM CHƠI TỰ ĐỘNG (PLAY TỪ FILE)
# ==========================================
def play():
    # Bật render_mode để xem AI chơi
    env = gym.make("FlappyBird-v0", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = FlappyAgent(state_dim, action_dim)
    
    model_file = "flappy_brain.pth"
    agent.load(model_file)

    print("🎮 AI ĐANG TỰ ĐỘNG CHƠI...")
    for e in range(5): # Chơi thử 5 mạng
        state, _ = env.reset()
        score = 0
        while True:
            # act với is_playing=True để dùng 100% kinh nghiệm, ko random
            action = agent.act(state, is_playing=True) 
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if reward == 1: # Qua được 1 ống là được 1 điểm
                score += 1
                
            state = next_state
            
            if done:
                print(f"💀 Mạng {e+1} kết thúc. Điểm qua ống: {score}")
                break
                
    env.close()

# ==========================================
# CHẠY CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Flappy Bird với Deep Q-Learning")
    parser.add_argument('--train', action='store_true', help="Huấn luyện mô hình mới và lưu file")
    parser.add_argument('--play', action='store_true', help="Tải file mô hình và xem AI tự chơi")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.play:
        play()
    else:
        print("⚠️ Vui lòng thêm cờ --train hoặc --play khi chạy. Ví dụ: python flappy_ai.py --train")