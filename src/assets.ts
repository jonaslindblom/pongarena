// Relative to Vite's base so the same build works at a domain root or /repo/.
export const assetPath = (path: string) => import.meta.env.BASE_URL + path.replace(/^\//, '')
