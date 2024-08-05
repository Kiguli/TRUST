<script setup>
defineProps({
    title: String,
});
</script>

<template>
    <div>
        <h2 v-if="title" class="my-1 text-sm font-medium text-gray-400">
            {{ title }}
        </h2>
        <pre
            class="mb-1 overflow-x-auto rounded-md bg-white p-2 text-sm"><slot /></pre>
    </div>
</template>
