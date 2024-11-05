<script setup>
import { computed } from "vue";

const props = defineProps({
    dimension: {
        type: Number,
        required: true,
    },
    monomials: {
        type: Array,
        required: true,
    },
});

const N = computed(() => props.monomials.length);

const model = defineModel();
</script>

<template>
    <slot name="title"></slot>
    <slot name="description"></slot>

    <!-- Create a text-area where the user can enter their (N, n) matrix of Theta(x) terms. -->
    <div class="mt-3">
        <textarea
            v-model="model.value"
            :rows="N"
            class="block min-h-24 w-full min-w-0 resize-y rounded-md border dark:border-transparent px-2 py-1.5 text-sm leading-6 placeholder:text-gray-400 dark:placeholder:text-gray-500 dark:bg-gray-950 outline-none focus:ring-2 ring-inset ring-purple-300"
            placeholder="Enter your matrix of &theta;(x) terms here..."></textarea>
    </div>
</template>

<style scoped></style>
