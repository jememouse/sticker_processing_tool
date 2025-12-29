<template>
  <div class="param-panel">
    <h3 class="panel-title">参数配置</h3>
    
    <!-- 白边宽度 -->
    <div class="param-group">
      <label class="param-label">
        白边宽度
        <span class="param-value">{{ outlineWidth }}px</span>
      </label>
      <input 
        type="range" 
        v-model.number="outlineWidth"
        min="5" 
        max="50" 
        step="1"
        class="param-slider"
      />
      <div class="slider-labels">
        <span>5px</span>
        <span>50px</span>
      </div>
    </div>
    
    <!-- AI 模型选择 -->
    <div class="param-group">
      <label class="param-label">AI 抠图模型</label>
      <div class="model-options">
        <button 
          class="model-btn"
          :class="{ active: modelType === 'birefnet' }"
          @click="modelType = 'birefnet'"
        >
          <span class="model-name">BiRefNet</span>
          <span class="model-desc">高精度</span>
        </button>
        <button 
          class="model-btn"
          :class="{ active: modelType === 'rembg' }"
          @click="modelType = 'rembg'"
        >
          <span class="model-name">Rembg</span>
          <span class="model-desc">更快速</span>
        </button>
      </div>
    </div>
    
    <!-- 生成按钮 -->
    <button 
      class="generate-btn"
      :disabled="!canGenerate || loading"
      @click="onGenerate"
    >
      <span v-if="loading" class="btn-loading"></span>
      <span v-else>生成贴纸</span>
    </button>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'

const props = defineProps({
  canGenerate: {
    type: Boolean,
    default: false
  },
  loading: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['generate'])

const outlineWidth = ref(15)
const modelType = ref('birefnet')

// 监听参数变化，通知父组件
watch([outlineWidth, modelType], () => {
  // 可以在这里触发预览更新
})

function onGenerate() {
  emit('generate', {
    outline_width: outlineWidth.value,
    model_type: modelType.value
  })
}
</script>

<style scoped>
.param-panel {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 16px;
  padding: 24px;
}

.panel-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--color-text);
  margin-bottom: 24px;
}

.param-group {
  margin-bottom: 24px;
}

.param-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.875rem;
  color: var(--color-text-secondary);
  margin-bottom: 12px;
}

.param-value {
  font-weight: 600;
  color: var(--color-primary);
}

.param-slider {
  width: 100%;
  height: 6px;
  border-radius: 3px;
  background: rgba(255, 255, 255, 0.1);
  appearance: none;
  cursor: pointer;
}

.param-slider::-webkit-slider-thumb {
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--color-primary);
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4);
}

.slider-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
  color: var(--color-text-muted);
  margin-top: 8px;
}

.model-options {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.model-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 16px;
  border-radius: 12px;
  border: 2px solid rgba(255, 255, 255, 0.1);
  background: transparent;
  cursor: pointer;
  transition: all 0.3s ease;
}

.model-btn:hover {
  border-color: rgba(255, 255, 255, 0.3);
}

.model-btn.active {
  border-color: var(--color-primary);
  background: rgba(99, 102, 241, 0.1);
}

.model-name {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--color-text);
  margin-bottom: 4px;
}

.model-desc {
  font-size: 0.75rem;
  color: var(--color-text-muted);
}

.generate-btn {
  width: 100%;
  padding: 16px;
  border-radius: 12px;
  border: none;
  background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
  color: white;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.generate-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
}

.generate-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-loading {
  width: 20px;
  height: 20px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
