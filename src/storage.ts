export interface SavedModel {
  id: string
  name: string
  bytes: Uint8Array
}
function database(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    // Keep the existing storage namespace so saved data survives the product rename.
    const req = indexedDB.open('rally-lab-models', 1)
    req.onupgradeneeded = () => req.result.createObjectStore('models', { keyPath: 'id' })
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}
export async function savedModels(): Promise<SavedModel[]> {
  const db = await database()
  try {
    return await new Promise((resolve, reject) => {
      const req = db.transaction('models').objectStore('models').getAll()
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error)
    })
  } finally {
    db.close()
  }
}
export async function saveModel(model: SavedModel) {
  const db = await database()
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction('models', 'readwrite')
      tx.objectStore('models').put(model)
      tx.oncomplete = () => resolve()
      tx.onabort = tx.onerror = () => reject(tx.error)
    })
  } finally {
    db.close()
  }
}
