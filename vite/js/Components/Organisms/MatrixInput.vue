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

const parseMatrix = (input) => {
    const matrix = input
        .trim()
        .split("\n")
        .map((row) => {
            return row
                .split(",")
                .map((item) => item.trim())
                .filter((value) => value !== undefined);
        });

    model.value = matrix;
};
</script>

<template>
    <slot name="title"></slot>
    <slot name="description"></slot>

    <div class="mt-3">
        <textarea
            :rows="N"
            class="block min-h-24 w-full min-w-0 resize-y rounded-md border px-2 py-1.5 text-sm leading-6 outline-none ring-inset ring-purple-300 placeholder:text-gray-400 focus:ring-2 dark:border-transparent dark:bg-gray-950 dark:placeholder:text-gray-500"
            placeholder="Enter your matrix of &theta;(x) terms here..."
            @input="parseMatrix($event.target.value)"></textarea>
    </div>
</template>

<style scoped></style>
