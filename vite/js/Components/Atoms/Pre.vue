<script setup>
import {
    ClipboardDocumentIcon,
    ClipboardDocumentCheckIcon,
} from "@heroicons/vue/16/solid";
import { useClipboard } from "@vueuse/core";
import { ref, useSlots } from "vue";

defineProps({
    title: String,
});

const slots = useSlots();

const source = ref();

const { text, copy, copied, isSupported } = useClipboard({ source });

const copyEvent = () => {
    copy(slots.default()[0].children);
};
</script>

<template>
    <div>
        <h2 v-if="title" class="my-1 text-sm text-gray-400">
            {{ title }}
        </h2>
        <div class="relative">
            <pre
                class="relative mb-1 max-h-36 overflow-auto whitespace-pre-line rounded-md bg-white p-2 text-sm dark:bg-gray-900/50 dark:text-gray-100"><slot
                ref="content" class="pointer-events-none" />
            </pre>
            <span
                v-if="isSupported"
                class="absolute right-2.5 top-2.5 cursor-pointer text-xs text-gray-400 hover:text-gray-600 dark:text-gray-300 dark:hover:text-gray-200"
                @click="copyEvent">
                <ClipboardDocumentIcon v-if="!copied" class="size-4" />
                <ClipboardDocumentCheckIcon v-else class="size-4" />
            </span>
        </div>
    </div>
</template>
