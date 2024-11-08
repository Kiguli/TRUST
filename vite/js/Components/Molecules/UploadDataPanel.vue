<script setup>
defineProps({
    form: {
        type: Object,
        required: true,
    },
    description: {
        type: String,
        default: "Drag and drop or select a file to upload.",
    },
});

const model = defineModel({ type: Object, default: () => ({}) });

const uploadFile = (file) => {
    model.value = file;
}
</script>

<template>
    <div
        class="mt-2 flex flex-col items-center rounded-lg border-2 border-dashed border-gray-300 p-4 focus:border-blue-300 focus:outline-none dark:border-gray-600">
        <div class="pt-2 pb-5 text-center text-sm">
            <p class="mb-1 font-medium text-gray-900 dark:text-gray-200">
                Choose a file.
            </p>
            <p class="text-gray-400">
                {{ description }}
            </p>
        </div>

        <input
            :class="[
                'file:m-1 file:mr-4 file:cursor-pointer file:rounded-md file:border-none file:bg-violet-900/75 file:px-5 file:py-3 file:text-white',
                'file:hover:bg-violet-900/80 file:focus:ring-2 file:focus:ring-white/60 file:focus:ring-offset-2 file:focus:ring-offset-violet-300',
                'file:active:ring-2 file:active:ring-white/60 file:active:ring-offset-2 file:active:ring-offset-violet-300',
            ]"
            class="block w-full rounded-lg border border-gray-300 bg-gray-100 text-sm text-gray-500 focus:outline-none dark:border-none dark:bg-gray-950"
            required
            type="file"
            @input="model = $event.target.files[0]" />
        <progress v-if="form?.progress" :value="form.progress.percentage" max="100">
            {{ form.progress.percentage }}
        </progress>
    </div>
</template>

<style scoped></style>
