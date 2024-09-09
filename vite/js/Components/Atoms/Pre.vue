<script setup>
import { onMounted, ref, useSlots } from "vue";
import {
    ClipboardDocumentIcon,
    ClipboardDocumentCheckIcon,
} from "@heroicons/vue/16/solid";
import { useClipboard } from "@vueuse/core";

defineProps({
    title: String,
});

const slots = useSlots();

const content = ref();
const { text, copy, copied, isSupported } = useClipboard({ content });
</script>

<template>
    <div class="relative">
        <h2 v-if="title" class="my-1 text-sm text-gray-400">
            {{ title }}
        </h2>
        <p class="text-white">
            Supported: {{ isSupported }}
        </p>
        <p class="text-white">
            Text: {{ text }}
        </p>
        <p class="text-white">
            Content: {{ content }}
        </p>
        <p class="text-white">
            Slot: {{ slots.default }}
        </p>
        <pre
            class="relative mb-1 overflow-x-auto rounded-md bg-white p-2 text-sm dark:bg-gray-900/50 dark:text-gray-100"><span
            v-if="isSupported"
            class="cursor-pointer absolute top-0 right-0 text-xs text-gray-400 hover:text-gray-600 dark:text-gray-300 dark:hover:text-gray-200"
            @click="copy(content)"><ClipboardDocumentIcon v-if="!copied" class="size-4" /><ClipboardDocumentCheckIcon
            v-else class="size-6" /></span><slot ref="content" /></pre>
    </div>
</template>
