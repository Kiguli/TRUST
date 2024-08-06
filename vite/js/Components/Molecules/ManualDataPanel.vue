<script setup>
const data = defineModel()
</script>

<template>
    <div
        class="flex h-56 flex-col rounded-lg border-2 border-dashed border-gray-300 p-4 dark:border-gray-600">
        <label for="dataset" class="pb-2 text-sm font-medium text-gray-900 dark:text-gray-200">
            Enter your dataset.
        </label>
        <p class="text-sm text-gray-400 pb-2">
            Each row should be a data point, with columns separated by commas.
        </p>

        <textarea
            v-model="data"
            rows="5"
            name="dataset"
            id="dataset"
            autocapitalize="off"
            autocomplete="off"
            aria-autocomplete="none"
            placeholder="e.g. [[1, 2], [3, 4]]"
            class="block w-full rounded-lg border-0 outline-none px-2 font-mono py-1.5 text-sm text-gray-900 dark:text-gray-100 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-violet-600 dark:bg-gray-950 dark:ring-0" />
    </div>
</template>

<style scoped></style>
