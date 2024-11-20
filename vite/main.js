import './css/app.css';

import { createApp, h } from 'vue';
import { createInertiaApp } from '@inertiajs/vue3';

import { resolvePageComponent } from './js/utilities/inertia-helpers.js';

import AppLayout from '@/Layouts/AppLayout.vue';

const appName = window.document.getElementsByTagName('title')[0]?.innerText || 'TRUST'

createInertiaApp({
    title: title => title.trim() === '' ? appName : `${title} - ${appName}`,
    resolve: name => {
        const page = resolvePageComponent(
            `./js/Components/Pages/${name}.vue`,
            import.meta.glob('./js/Components/Pages/**/*.vue', { eager: true }),
        );

        page.then(module => {
            const page = module.default;
            let layout = page.layout;

            if (layout === undefined) {
                layout = AppLayout;
            }

            page.layout = layout;
        });

        return page;
    },
    setup ({ el, App, props, plugin }) {
        createApp({ render: () => h(App, props) }).use(plugin).mount(el);
    },
});