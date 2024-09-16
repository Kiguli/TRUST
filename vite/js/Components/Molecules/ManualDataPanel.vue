<script setup>
import { ref, watchEffect } from "vue";

const data = defineModel();
const userData = ref("");

const inputError = ref(false);

watchEffect(() => {
    inputError.value = false;

    if (!userData.value) return;

    const parsed = userData.value
        // split the input by newlines or by comma-newlines
        .split(/\n|,\n/)
        // Remove any whitespace within a row
        .map((row) => row.replace(/\s/g, ""))
        // split each row by commas
        .map((row) => row.split(","))
        // remove empty values
        .map((row) => row.filter((col) => col !== ""));

    // If any row has a different number of columns, show an error
    const columns = parsed.map((row) => row.length);
    if (columns.some((col) => col !== columns[0])) {
        inputError.value = true;
        console.error("Invalid dataset: rows have different numbers of columns.");
        return;
    }

    data.value = parsed;
});
</script>

<template>
    <div
        class="flex flex-col rounded-lg border-2 border-dashed border-gray-300 p-4 dark:border-gray-600">
        <label class="pb-2 text-sm font-medium text-gray-900 dark:text-gray-200" for="dataset">
            Enter your dataset.
        </label>
        <p class="text-sm text-gray-400 pb-2">
            Each row should be a data point, with columns separated by commas.
        </p>

        <textarea
            id="dataset"
            v-model="userData"
            :class="[
                inputError ? 'ring-inset ring-2 ring-red-700' : 'focus:ring-violet-600 dark:ring-0',
                'ring-1 ring-inset ring-gray-300 focus:ring-2 focus:ring-inset'
            ]"
            aria-autocomplete="none"
            autocapitalize="off"
            autocomplete="off"
            class="block min-h-24 w-full rounded-lg border-0 outline-none px-2 font-mono py-1.5 text-sm text-gray-900 dark:text-gray-100 shadow-sm placeholder:text-gray-400 dark:bg-gray-950"
            name="dataset"
            placeholder="e.g. 1, 2, 3"
            required
            rows="5" />

        <p>
            <small class="text-gray-400 text-sm">
                Dimensions: {{ data[0]?.length || 1 }}
            </small>
        </p>
    </div>
</template>

<style scoped></style>
