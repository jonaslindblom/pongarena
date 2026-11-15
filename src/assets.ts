// Each document's <base> points to the site root, including nested reading pages.
export const assetPath = (path: string) => './' + path.replace(/^\//, '')
