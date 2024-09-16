<script setup>
import { LinkIcon } from "@heroicons/vue/16/solid/index.js";

import H3 from "@/Atoms/H3.vue";
import { onMounted, ref } from "vue";

defineProps({
    text: String,
    form: Object,
});

const submitBtn = ref();

onMounted(() => {
    window.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
            submitBtn.value?.click();
        }
    });
});

</script>

<template>
    <div
        class="sticky bottom-0 col-span-full flex min-h-20 w-full justify-between gap-x-4 border-t bg-white px-4 py-2.5 sm:px-8 dark:border-none dark:bg-gray-900 dark:shadow-inner">
        <div class="flex flex-col justify-center">
            <h3
                class="mb-0.5 text-base font-medium text-gray-800 dark:text-gray-200">
                <a
                    class="flex w-min cursor-pointer items-center gap-x-0.5"
                    href="https://github.com/kiguli/sintrajbc"
                    target="_blank">
                    <span class="inline-block flex-none">SinTra-SB</span>
                    <LinkIcon class="h-4 w-4 flex-none text-gray-500" />
                </a>
            </h3>
            <p class="line-clamp-2 text-xs text-gray-400">
                Single Trajectory Data-Driven Control Synthesis for
                Stability and Barrier Certificates
            </p>
        </div>
        <div class="flex items-center gap-x-2">
            <button
                ref="submitBtn"
                :disabled="form.processing"
                class="order-1 flex items-baseline gap-x-1 h-min rounded-md bg-blue-600/75 px-4 py-2 text-base text-gray-50 outline-none hover:bg-blue-700/75 focus:ring-blue-600 active:bg-blue-800/75 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-blue-600/75 sm:px-5 sm:py-2.5"
                type="submit"
                v-html="text" />
            <button
                class="order-0 flex h-min rounded-md px-4 py-2 text-base text-gray-400 hover:ring-2 hover:ring-inset hover:ring-gray-400/75 active:text-gray-200 active:ring-gray-300 sm:px-5 sm:py-2.5">
                Reset
            </button>
        </div>
    </div>
</template>

<style scoped>

</style>