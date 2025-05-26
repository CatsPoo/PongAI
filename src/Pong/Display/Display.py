import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # headless backend
import matplotlib.pyplot as plt

class Display:
    def __init__(self,window_hight,window_width,dpi,show_graph,fps):
        self.width = window_width
        self.height = window_hight
        self.show_graph = show_graph 
        self.fps = fps
        self.dpi = dpi
        self.left_agent_rewards = []
        self.right_agent_rewards = []
        self.epsilons = []
        self.graph_frame = self.get_progress_plot()

    def update(self,game,left_agent_rewards,right_agent_rewards,epsilons):
        if(not np.array_equal(self.left_agent_rewards, left_agent_rewards) or not np.array_equal(self.right_agent_rewards, right_agent_rewards) or not np.array_equal(self.epsilons, epsilons)):
            self.left_agent_rewards = left_agent_rewards.copy()
            self.right_agent_rewards = right_agent_rewards.copy()
            self.epsilons = epsilons.copy()
            self.graph_frame = self.get_progress_plot()

        image_to_display = self.concat_side_by_side(game,self.graph_frame)
        cv2.imshow('Game',image_to_display)
        cv2.waitKey(1000//self.fps)

    def get_progress_plot(self) -> None:
        if not (len(self.left_agent_rewards) == len(self.right_agent_rewards) == len(self.epsilons)):
            raise ValueError("all three sequences must be the same length")

        figsize = (self.width /2 / self.dpi, self.height / self.dpi)  # inches
        ep_idx  = np.arange(len(self.left_agent_rewards))

        # ── build the figure ───────────────────────────────────────────────
        fig, ax1 = plt.subplots(figsize=figsize, dpi=self.dpi)

        ax1.plot(ep_idx, self.left_agent_rewards,  color="tab:blue",  label="Left reward")
        ax1.plot(ep_idx, self.right_agent_rewards, color="tab:red", label="Right reward")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="upper left")

        # epsilon on a second Y axis
        ax2 = ax1.twinx()
        ax2.plot(ep_idx, self.epsilons, color="tab:green", label="Epsilon")
        ax2.set_ylabel("Epsilon", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
        ax2.legend(loc="upper right")

        fig.tight_layout(pad=1.2)

        # ── render → NumPy array ───────────────────────────────────────────
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf  = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_rgb = buf.reshape((h, w, 4))[:, :, :3].copy()  # drop alpha

        plt.close(fig)
        return img_rgb
    
    def concat_side_by_side(self,img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
        h1, w1 = img_left.shape[:2]
        h2, w2 = img_right.shape[:2]

        # --- resize right image to match left height if needed ---
        if h1 != h2:
            scale = h1 / h2
            img_right = cv2.resize(img_right, (int(w2 * scale), h1),
                                interpolation=cv2.INTER_AREA)

        # --- concatenate on the width axis ---
        return np.hstack([img_left, img_right])
