<template>
  <div 
    class="file-upload"
    :class="{ 'is-dragging': isDragging }"
    @dragover.prevent="onDragOver"
    @dragleave.prevent="onDragLeave"
    @drop.prevent="onDrop"
    @click="triggerInput"
  >
    <input 
      ref="fileInput"
      type="file"
      accept="image/jpeg,image/png,image/webp"
      @change="onFileChange"
      hidden
    />
    
    <div class="upload-content">
      <div class="upload-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
      </div>
      <p class="upload-text">拖拽图片到此处</p>
      <p class="upload-hint">或点击选择文件</p>
      <p class="upload-formats">支持 JPG、PNG、WebP（最大 20MB）</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const emit = defineEmits(['file-selected'])

const fileInput = ref(null)
const isDragging = ref(false)

function triggerInput() {
  fileInput.value?.click()
}

function onDragOver() {
  isDragging.value = true
}

function onDragLeave() {
  isDragging.value = false
}

function onDrop(e) {
  isDragging.value = false
  const files = e.dataTransfer?.files
  if (files && files.length > 0) {
    handleFile(files[0])
  }
}

function onFileChange(e) {
  const files = e.target?.files
  if (files && files.length > 0) {
    handleFile(files[0])
  }
}

function handleFile(file) {
  // 验证文件类型
  const allowedTypes = ['image/jpeg', 'image/png', 'image/webp']
  if (!allowedTypes.includes(file.type)) {
    alert('不支持的文件格式，请选择 JPG、PNG 或 WebP 图片')
    return
  }
  
  // 验证文件大小 (20MB)
  if (file.size > 20 * 1024 * 1024) {
    alert('文件大小超过 20MB 限制')
    return
  }
  
  emit('file-selected', file)
}
</script>

<style scoped>
.file-upload {
  border: 2px dashed rgba(255, 255, 255, 0.3);
  border-radius: 16px;
  padding: 48px 24px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: rgba(255, 255, 255, 0.05);
}

.file-upload:hover,
.file-upload.is-dragging {
  border-color: var(--color-primary);
  background: rgba(99, 102, 241, 0.1);
}

.upload-content {
  pointer-events: none;
}

.upload-icon {
  width: 64px;
  height: 64px;
  margin: 0 auto 16px;
  color: var(--color-primary);
}

.upload-icon svg {
  width: 100%;
  height: 100%;
}

.upload-text {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--color-text);
  margin-bottom: 8px;
}

.upload-hint {
  color: var(--color-text-secondary);
  margin-bottom: 16px;
}

.upload-formats {
  font-size: 0.875rem;
  color: var(--color-text-muted);
}
</style>
