/**
 * 贴纸 API 调用封装
 */
import axios from 'axios'

// API 基础地址 (使用 Vite 代理时为空，生产环境可配置)
const API_BASE = ''

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000, // 60秒超时 (模型处理可能较慢)
})

/**
 * 上传图片
 * @param {File} file - 图片文件
 * @returns {Promise<{file_id: string, filename: string, preview_url: string}>}
 */
export async function uploadImage(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await api.post('/api/sticker/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

/**
 * 生成贴纸
 * @param {Object} params - 生成参数
 * @param {string} params.file_id - 上传文件 ID
 * @param {number} params.outline_width - 白边宽度
 * @param {string} params.model_type - AI 模型类型 (birefnet / rembg)
 * @returns {Promise<{task_id: string, status: string}>}
 */
export async function generateSticker(params) {
  const response = await api.post('/api/sticker/generate', params)
  return response.data
}

/**
 * 获取任务状态
 * @param {string} taskId - 任务 ID
 * @returns {Promise<{task_id: string, status: string, progress: number, preview_url: string, pdf_url: string, svg_url: string}>}
 */
export async function getTaskStatus(taskId) {
  const response = await api.get(`/api/sticker/status/${taskId}`)
  return response.data
}

/**
 * 获取下载链接
 * @param {string} taskId - 任务 ID
 * @param {string} fileType - 文件类型 (preview / pdf / svg)
 * @returns {string} 下载 URL
 */
export function getDownloadUrl(taskId, fileType) {
  return `${API_BASE}/api/sticker/download/${taskId}/${fileType}`
}

/**
 * 获取原图预览链接
 * @param {string} fileId - 文件 ID
 * @returns {string} 预览 URL
 */
export function getOriginalPreviewUrl(fileId) {
  return `${API_BASE}/api/sticker/preview/original/${fileId}`
}

export default api
