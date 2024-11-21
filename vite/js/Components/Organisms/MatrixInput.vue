<script setup>
import { computed, ref, watch } from "vue";

const props = defineProps({
    dimension: {
        type: Number,
        required: true,
    },
    monomials: {
        type: Object,
        required: true,
    },
});

const N = computed(() => props.monomials.length);

const model = defineModel();
</script>

<template>
    <div class="relative">
        <slot name="title"></slot>
    </div>
    <slot name="description"></slot>

    <div class="mt-3">
        <textarea
            :rows="N"
            class="block min-h-24 w-full min-w-0 resize-y rounded-md border px-2 py-1.5 text-sm leading-6 outline-none ring-inset ring-purple-300 placeholder:text-gray-400 focus:ring-2 dark:border-transparent dark:bg-gray-950 dark:placeholder:text-gray-500"
            placeholder="Enter your matrix of &theta;(x) terms here..."
            v-model="model"></textarea>
    </div>
</template>

<style scoped></style>
