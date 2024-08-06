<script setup>
defineProps({
    title: String,
});
</script>

<template>
    <div>
        <h2 v-if="title" class="my-1 text-sm text-gray-400">
            {{ title }}
        </h2>
        <pre
            class="mb-1 overflow-x-auto rounded-md bg-white p-2 text-sm dark:bg-gray-900/50 dark:text-gray-100"><slot /></pre>
    </div>
</template>
