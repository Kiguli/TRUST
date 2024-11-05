<script setup>
import { computed } from "vue";

const props = defineProps({
    dimension: {
        type: Number,
        required: true
    },
    monomials: {
        type: Array,
        required: true
    },
})

const N = computed(() => props.monomials.length)

const model = defineModel()
</script>

<template>
    <slot name="title"></slot>
    <slot name="description"></slot>
</template>

<style scoped></style>
