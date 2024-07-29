<script setup>
import {
    RadioGroup,
    RadioGroupDescription,
    RadioGroupLabel,
    RadioGroupOption,
} from "@headlessui/vue"
import { ref } from "vue"

const props = defineProps({
    classes: Array,
})

const selected = ref(props.classes[0].title)
</script>

<template>
    <div>
        <h2 class="my-2 text-xl font-semibold text-gray-700">Problem Class</h2>

        <RadioGroup v-model="selected">
            <RadioGroupLabel class="sr-only">Problem Class</RadioGroupLabel>
            <div class="space-y-2">
                <RadioGroupOption
                    as="template"
                    v-for="classType in classes"
                    :key="classType.title"
                    :value="classType.title"
                    v-slot="{ active, checked }">
                    <div
                        class="relative flex cursor-pointer rounded-lg px-5 py-4 shadow-md focus:outline-none"
                        :class="[
                            active
                                ? 'ring-2 ring-white/60 ring-offset-2 ring-offset-indigo-300'
                                : '',
                            checked ? 'bg-indigo-900/75 text-white' : 'bg-white',
                        ]">
                        <div class="flex w-full items-center justify-between">
                            <div class="flex items-center">
                                <div class="text-sm">
                                    <RadioGroupLabel
                                        as="p"
                                        class="font-medium"
                                        :class="
                                            checked
                                                ? 'text-white'
                                                : 'text-gray-900'
                                        ">
                                        {{ classType.title }}
                                    </RadioGroupLabel>

                                    <RadioGroupDescription
                                        as="span"
                                        class="inline"
                                        :class="
                                            checked
                                                ? 'text-indigo-100'
                                                : 'text-gray-500'
                                        ">
                                        <span>{{ classType.description }}</span>
                                    </RadioGroupDescription>
                                </div>
                            </div>

                            <div v-show="checked" class="shrink-0 text-white">
                                <svg
                                    class="h-6 w-6"
                                    viewBox="0 0 24 24"
                                    fill="none">
                                    <circle
                                        cx="12"
                                        cy="12"
                                        r="12"
                                        fill="#fff"
                                        fill-opacity="0.2" />
                                    <path
                                        d="M7 13l3 3 7-7"
                                        stroke="#fff"
                                        stroke-width="1.5"
                                        stroke-linecap="round"
                                        stroke-linejoin="round" />
                                </svg>
                            </div>
                        </div>
                    </div>
                </RadioGroupOption>
            </div>
        </RadioGroup>
    </div>
</template>

<style scoped></style>
