<template>
  <div class="download-bar">
    <h4 class="bar-title">下载文件</h4>
    
    <div class="download-buttons">
      <button 
        class="download-btn"
        :disabled="!taskCompleted"
        @click="download('preview')"
      >
        <span class="btn-icon">🖼️</span>
        <span class="btn-label">预览 PNG</span>
      </button>
      
      <button 
        class="download-btn primary"
        :disabled="!taskCompleted"
        @click="download('pdf')"
      >
        <span class="btn-icon">📄</span>
        <span class="btn-label">印刷 PDF</span>
      </button>
      
      <button 
        class="download-btn"
        :disabled="!taskCompleted"
        @click="download('svg')"
      >
        <span class="btn-icon">✂️</span>
        <span class="btn-label">刀线 SVG</span>
      </button>
    </div>
    
    <p v-if="!taskCompleted" class="hint">
      请先上传图片并生成贴纸
    </p>
  </div>
</template>

<script setup>
import { getDownloadUrl } from '../api/sticker'

const props = defineProps({
  taskId: {
    type: String,
    default: ''
  },
  taskCompleted: {
    type: Boolean,
    default: false
  }
})

function download(fileType) {
  if (!props.taskId) return
  
  const url = getDownloadUrl(props.taskId, fileType)
  window.open(url, '_blank')
}
</script>

<style scoped>
.download-bar {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 16px;
  padding: 24px;
}

.bar-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--color-text);
  margin-bottom: 16px;
}

.download-buttons {
  display: flex;
  gap: 12px;
}

.download-btn {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 16px 12px;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.05);
  cursor: pointer;
  transition: all 0.3s ease;
}

.download-btn:hover:not(:disabled) {
  border-color: var(--color-primary);
  background: rgba(99, 102, 241, 0.1);
  transform: translateY(-2px);
}

.download-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.download-btn.primary {
  background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
  border: none;
}

.download-btn.primary:hover:not(:disabled) {
  box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
}

.btn-icon {
  font-size: 1.5rem;
}

.btn-label {
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--color-text);
}

.hint {
  margin-top: 16px;
  font-size: 0.875rem;
  color: var(--color-text-muted);
  text-align: center;
}
</style>
