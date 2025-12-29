<template>
  <div class="preview-area">
    <!-- 空状态 -->
    <div v-if="!imageUrl && !loading" class="empty-state">
      <div class="watermark">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
          <rect x="3" y="3" width="18" height="18" rx="2"/>
          <circle cx="8.5" cy="8.5" r="1.5"/>
          <path d="M21 15l-5-5L5 21"/>
        </svg>
        <p>上传图片开始制作贴纸</p>
      </div>
    </div>
    
    <!-- 加载状态 -->
    <div v-if="loading" class="loading-state">
      <div class="progress-ring">
        <svg viewBox="0 0 100 100">
          <circle class="progress-bg" cx="50" cy="50" r="40"/>
          <circle 
            class="progress-bar" 
            cx="50" cy="50" r="40"
            :style="{ strokeDashoffset: progressOffset }"
          />
        </svg>
        <span class="progress-text">{{ progress }}%</span>
      </div>
      <p class="loading-hint">{{ statusText }}</p>
    </div>
    
    <!-- 预览图片 -->
    <div v-if="imageUrl && !loading" class="preview-container">
      <img :src="imageUrl" :alt="imageAlt" class="preview-image" />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  imageUrl: {
    type: String,
    default: ''
  },
  imageAlt: {
    type: String,
    default: '预览'
  },
  loading: {
    type: Boolean,
    default: false
  },
  progress: {
    type: Number,
    default: 0
  },
  status: {
    type: String,
    default: ''
  }
})

// 进度条偏移计算 (圆周长 = 2 * π * r = 2 * 3.14159 * 40 ≈ 251)
const progressOffset = computed(() => {
  const circumference = 251
  return circumference - (props.progress / 100) * circumference
})

const statusText = computed(() => {
  switch (props.status) {
    case 'pending': return '准备中...'
    case 'processing': return '正在处理...'
    case 'completed': return '完成!'
    default: return '处理中...'
  }
})
</script>

<style scoped>
.preview-area {
  background: rgba(0, 0, 0, 0.3);
  border-radius: 16px;
  min-height: 400px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  position: relative;
}

.empty-state {
  text-align: center;
}

.watermark {
  color: rgba(255, 255, 255, 0.2);
}

.watermark svg {
  width: 80px;
  height: 80px;
  margin-bottom: 16px;
}

.watermark p {
  font-size: 1rem;
}

.loading-state {
  text-align: center;
}

.progress-ring {
  position: relative;
  width: 120px;
  height: 120px;
  margin: 0 auto 24px;
}

.progress-ring svg {
  width: 100%;
  height: 100%;
  transform: rotate(-90deg);
}

.progress-bg {
  fill: none;
  stroke: rgba(255, 255, 255, 0.1);
  stroke-width: 8;
}

.progress-bar {
  fill: none;
  stroke: var(--color-primary);
  stroke-width: 8;
  stroke-linecap: round;
  stroke-dasharray: 251;
  transition: stroke-dashoffset 0.3s ease;
}

.progress-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--color-text);
}

.loading-hint {
  color: var(--color-text-secondary);
}

.preview-container {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}

.preview-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  border-radius: 8px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
</style>
