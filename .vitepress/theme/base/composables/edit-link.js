import { computed } from 'vue';
import { useData } from './data.js';
export function useEditLink() {
    const { theme, page } = useData();
    return computed(() => {
        const { text = 'Edit this page', pattern = '' } = theme.value.editLink || {};
        const { relativePath } = page.value;
        let url;
        if (typeof pattern === 'function') {
            url = pattern({ relativePath });
        }
        else {
            url = pattern.replace(/:path/g, relativePath);
        }
        return { url, text };
    });
}
